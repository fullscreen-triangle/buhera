/**
 * GitHub fetch layer — pulls a repo's tree, source blobs, and git stats using a
 * user-supplied personal-access token. Runs entirely in the browser; the token
 * never leaves the client except to api.github.com.
 *
 * This is the *fetching* half of the browser tool. The *analysis* half (χ, sense)
 * lives in `chi.ts` and consumes the `RepoSnapshot` produced here.
 */

const GH = "https://api.github.com";

/** File extensions we treat as indexable source (mirrors the Rust `purpose` set). */
const INDEXABLE = new Set([
  "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp",
  "cs", "rb", "php", "swift", "kt", "scala", "md", "tex",
]);

/** Path components skipped during indexing (mirrors `purpose`'s ignore list). */
const IGNORED = new Set([
  ".git", "node_modules", "target", ".purpose", "dist", "build", "__pycache__",
  ".venv", "venv", ".next", ".nuxt", ".cache", "coverage", "out", "vendor",
  ".idea", ".vscode",
]);

export interface RepoRef {
  owner: string;
  name: string;
}

/** One fetched source file with its text. */
export interface SourceFile {
  path: string;
  ext: string;
  text: string;
  size: number;
}

/** Everything the analysis needs about one repo, fetched from GitHub. */
export interface RepoSnapshot {
  ref: RepoRef;
  defaultBranch: string;
  description: string | null;
  language: string | null;
  topics: string[];
  stars: number;
  openIssues: number;
  pushedAt: string | null;
  files: SourceFile[];
  /** Files present in the tree but not fetched (too big / too many / non-source). */
  skipped: number;
  /** Recent commit activity: total commits in the sampled window. */
  commitCount: number;
  contributors: { login: string; contributions: number }[];
  /** Non-fatal notes surfaced to the UI (rate limits, truncation, etc.). */
  notes: string[];
}

export class GitHubError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "GitHubError";
  }
}

/** Parse "owner/name", a full URL, or "https://github.com/owner/name(.git)". */
export function parseRepoRef(input: string): RepoRef {
  const trimmed = input.trim().replace(/\.git$/, "");
  const urlMatch = trimmed.match(/github\.com[/:]([^/]+)\/([^/]+)/i);
  if (urlMatch) return { owner: urlMatch[1], name: urlMatch[2] };
  const parts = trimmed.split("/").filter(Boolean);
  if (parts.length >= 2) return { owner: parts[parts.length - 2], name: parts[parts.length - 1] };
  throw new Error(`could not parse a repo from ${JSON.stringify(input)} (expected owner/name or a GitHub URL)`);
}

interface FetchOpts {
  token: string;
  /** Max source files to fetch (guards rate limits). */
  maxFiles?: number;
  /** Max bytes per file to fetch. */
  maxFileBytes?: number;
  /** Progress callback: (done, total, label). */
  onProgress?: (done: number, total: number, label: string) => void;
}

async function ghJson<T>(path: string, token: string): Promise<T> {
  const res = await fetch(`${GH}${path}`, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    if (res.status === 401) throw new GitHubError(401, "token rejected (401) — check the personal-access token has repo read scope");
    if (res.status === 403) throw new GitHubError(403, "forbidden / rate-limited (403). Wait, or use a token with more quota.");
    if (res.status === 404) throw new GitHubError(404, "repo not found (404) — private repos need a token with access to them");
    throw new GitHubError(res.status, `GitHub API ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

function extOf(path: string): string {
  const dot = path.lastIndexOf(".");
  return dot === -1 ? "" : path.slice(dot + 1).toLowerCase();
}

function isIgnored(path: string): boolean {
  return path.split("/").some((c) => IGNORED.has(c));
}

/** Fetch a full repo snapshot: metadata, source blobs, and git stats. */
export async function fetchRepo(ref: RepoRef, opts: FetchOpts): Promise<RepoSnapshot> {
  const { token } = opts;
  const maxFiles = opts.maxFiles ?? 400;
  const maxFileBytes = opts.maxFileBytes ?? 120_000;
  const notes: string[] = [];
  const report = opts.onProgress ?? (() => {});

  report(0, 1, "reading repo metadata");
  const meta = await ghJson<any>(`/repos/${ref.owner}/${ref.name}`, token);
  const defaultBranch: string = meta.default_branch ?? "main";

  report(0, 1, "reading file tree");
  const tree = await ghJson<any>(
    `/repos/${ref.owner}/${ref.name}/git/trees/${defaultBranch}?recursive=1`,
    token
  );
  if (tree.truncated) notes.push("file tree was truncated by GitHub; analysis covers a partial tree");

  const blobs: { path: string; ext: string; sha: string; size: number }[] = (tree.tree ?? [])
    .filter((n: any) => n.type === "blob")
    .map((n: any) => ({ path: n.path, ext: extOf(n.path), sha: n.sha, size: n.size ?? 0 }))
    .filter((b: any) => INDEXABLE.has(b.ext) && !isIgnored(b.path) && b.size <= maxFileBytes);

  const totalIndexable = blobs.length;
  const toFetch = blobs.slice(0, maxFiles);
  const skipped = totalIndexable - toFetch.length;
  if (skipped > 0) notes.push(`fetched ${toFetch.length} of ${totalIndexable} indexable files (cap ${maxFiles})`);

  const files: SourceFile[] = [];
  let done = 0;
  // Fetch blobs with bounded concurrency to respect rate limits.
  const CONCURRENCY = 8;
  for (let i = 0; i < toFetch.length; i += CONCURRENCY) {
    const batch = toFetch.slice(i, i + CONCURRENCY);
    const results = await Promise.all(
      batch.map(async (b) => {
        try {
          const blob = await ghJson<any>(`/repos/${ref.owner}/${ref.name}/git/blobs/${b.sha}`, token);
          const text = blob.encoding === "base64" ? decodeBase64(blob.content) : (blob.content ?? "");
          return { path: b.path, ext: b.ext, text, size: b.size } as SourceFile;
        } catch {
          return null;
        }
      })
    );
    for (const r of results) if (r) files.push(r);
    done += batch.length;
    report(done, toFetch.length, "fetching source");
  }

  // Git stats (best-effort; failures are non-fatal notes).
  let commitCount = 0;
  let contributors: { login: string; contributions: number }[] = [];
  try {
    report(0, 1, "reading contributors");
    const contribs = await ghJson<any[]>(
      `/repos/${ref.owner}/${ref.name}/contributors?per_page=10`,
      token
    );
    contributors = (contribs ?? []).map((c) => ({ login: c.login, contributions: c.contributions }));
    commitCount = contributors.reduce((s, c) => s + c.contributions, 0);
  } catch {
    notes.push("contributor stats unavailable");
  }

  return {
    ref,
    defaultBranch,
    description: meta.description ?? null,
    language: meta.language ?? null,
    topics: meta.topics ?? [],
    stars: meta.stargazers_count ?? 0,
    openIssues: meta.open_issues_count ?? 0,
    pushedAt: meta.pushed_at ?? null,
    files,
    skipped,
    commitCount,
    contributors,
    notes,
  };
}

/** Base64 → UTF-8 text, browser-safe. */
function decodeBase64(b64: string): string {
  const clean = b64.replace(/\n/g, "");
  const bytes = Uint8Array.from(atob(clean), (c) => c.charCodeAt(0));
  return new TextDecoder("utf-8").decode(bytes);
}
