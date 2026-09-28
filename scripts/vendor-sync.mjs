#!/usr/bin/env node
// vendor-sync — keep vendored engine copies byte-identical to their upstream
// repositories at a recorded commit (specification 06-sourcing.md).
//
//   node scripts/vendor-sync.mjs               # --check every entry (default)
//   node scripts/vendor-sync.mjs --check [id…]
//   node scripts/vendor-sync.mjs --sync  <id…|all>   # re-copy from upstream HEAD, record commit
//   node scripts/vendor-sync.mjs --list
//
// Manifest: specifications/registry/vendor.json. Upstream content is read with
// `git show <commit>:<path>` from a local clone — never from a working tree, so
// uncommitted upstream edits are never vendored. Clone locations come from the
// manifest's `clones` map (paths relative to this repo's root), overridable per
// repo with BUHERA_CLONE_<REPO> (e.g. BUHERA_CLONE_HEGEL=/src/hegel).
//
// Entry fields: id, repo, from, to, commit, mode ("tree" | "file" | "build"),
// and optionally `exclude` (upstream paths under `from` deliberately not
// vendored; trailing "/" = a directory) and `local` (vendored paths that carry
// recorded local changes and are skipped by the byte check — each must be
// justified in the entry's `note`). "build" entries vendor a build product
// (e.g. compiled dist/): they cannot be byte-verified, so --check reports only
// whether the upstream source under `from` moved, and --sync refuses.
// Text is compared with CRLF normalised to LF (git autocrlf checkouts).
//
// --check exit status: 0 all verified, 1 an integrity failure (vendored bytes ≠
// upstream at the recorded commit), 2 manifest/usage error. "upstream moved"
// (HEAD differs from the recorded commit under the vendored path) is reported
// but is not a failure: it means a --sync is available, not that anything broke.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST = path.join(ROOT, "specifications/registry/vendor.json");
const CATALOGUE = path.join(ROOT, "specifications/registry/catalogue.json");

function die(msg) {
  console.error(`vendor-sync: ${msg}`);
  process.exit(2);
}

function git(cwd, args, opts = {}) {
  return execFileSync("git", args, { cwd, maxBuffer: 256 * 1024 * 1024, ...opts });
}

function cloneFor(manifest, repo) {
  const name = repo.split("/").pop().toUpperCase().replace(/[^A-Z0-9]/g, "_");
  const env = process.env[`BUHERA_CLONE_${name}`];
  const rel = env || manifest.clones[repo];
  if (!rel) return null;
  const dir = path.resolve(ROOT, rel);
  if (!existsSync(dir)) return null;
  // `from` paths are relative to the repository root, whatever subdirectory
  // the clone hint points at.
  return git(dir, ["rev-parse", "--show-toplevel"]).toString().trim();
}

/** Upstream files under `from` at `rev`: [{ rel, gitPath }]. */
function upstreamFiles(clone, rev, entry) {
  const { from, mode } = entry;
  if (mode === "file") return [{ rel: "", gitPath: from }];
  const out = git(clone, ["ls-tree", "-r", "--name-only", rev, "--", from]).toString();
  return out
    .split("\n")
    .filter(Boolean)
    .map((p) => ({ rel: p.slice(from.length + 1), gitPath: p }))
    .filter((f) => !excluded(entry, f.rel));
}

function excluded(entry, rel) {
  return (entry.exclude || []).some((x) => (x.endsWith("/") ? rel.startsWith(x) : rel === x));
}

const lf = (buf) => buf.toString("latin1").replace(/\r\n/g, "\n");

function vendoredFiles(dir) {
  const out = [];
  const walk = (d, pre) => {
    for (const e of readdirSync(d)) {
      const abs = path.join(d, e);
      const rel = pre ? `${pre}/${e}` : e;
      if (statSync(abs).isDirectory()) walk(abs, rel);
      else out.push(rel);
    }
  };
  if (existsSync(dir)) walk(dir, "");
  return out.sort();
}

function treeId(clone, rev, from) {
  try {
    return git(clone, ["rev-parse", `${rev}:${from}`], { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
  } catch {
    return null;
  }
}

function check(manifest, entry) {
  const clone = cloneFor(manifest, entry.repo);
  if (!clone) return { status: "unverifiable", detail: `no local clone of ${entry.repo}` };
  if (entry.mode === "build") {
    const moved = treeId(clone, entry.commit, entry.from) !== treeId(clone, "HEAD", entry.from);
    return moved
      ? { status: "moved", detail: `build product; upstream source under ${entry.from} changed (rebuild available)` }
      : { status: "built", detail: `build product of ${entry.from} @ ${entry.commit.slice(0, 7)} (source unchanged)` };
  }
  const to = path.join(ROOT, entry.to);
  const local = new Set(entry.local || []);
  const files = upstreamFiles(clone, entry.commit, entry).filter((f) => !local.has(f.rel));
  if (files.length === 0) return { status: "failed", detail: `nothing at ${entry.commit}:${entry.from}` };
  const problems = [];
  for (const f of files) {
    const dest = entry.mode === "file" ? to : path.join(to, f.rel);
    if (!existsSync(dest)) {
      problems.push(`missing ${f.rel || path.basename(to)}`);
      continue;
    }
    const want = git(clone, ["show", `${entry.commit}:${f.gitPath}`]);
    if (lf(want) !== lf(readFileSync(dest))) problems.push(`differs ${f.rel || path.basename(to)}`);
  }
  if (entry.mode !== "file") {
    const expected = new Set(files.map((f) => f.rel));
    for (const rel of vendoredFiles(to)) if (!expected.has(rel) && !local.has(rel)) problems.push(`extra ${rel}`);
  }
  if (problems.length) return { status: "failed", detail: problems.join("; ") };
  const moved = treeId(clone, entry.commit, entry.from) !== treeId(clone, "HEAD", entry.from);
  return moved
    ? { status: "moved", detail: `upstream HEAD differs under ${entry.from} (sync available)` }
    : { status: "verified", detail: `${files.length} file(s) @ ${entry.commit.slice(0, 7)}` };
}

function sync(manifest, entry) {
  const clone = cloneFor(manifest, entry.repo);
  if (!clone) die(`cannot sync ${entry.id}: no local clone of ${entry.repo}`);
  if (entry.mode === "build") die(`${entry.id} is a build product; rebuild it upstream and copy by hand (see its note)`);
  if ((entry.local || []).length) die(`${entry.id} carries local changes (${entry.local.join(", ")}); re-apply them by hand after syncing`);
  const head = git(clone, ["rev-parse", "HEAD"]).toString().trim();
  const files = upstreamFiles(clone, head, entry);
  if (files.length === 0) die(`cannot sync ${entry.id}: nothing at HEAD:${entry.from}`);
  const to = path.join(ROOT, entry.to);
  if (entry.mode === "file") {
    mkdirSync(path.dirname(to), { recursive: true });
    writeFileSync(to, git(clone, ["show", `HEAD:${entry.from}`]));
  } else {
    rmSync(to, { recursive: true, force: true });
    for (const f of files) {
      const dest = path.join(to, f.rel);
      mkdirSync(path.dirname(dest), { recursive: true });
      writeFileSync(dest, git(clone, ["show", `HEAD:${f.gitPath}`]));
    }
  }
  entry.commit = head;
  return { status: "synced", detail: `${files.length} file(s) @ ${head.slice(0, 7)}` };
}

/** Propagate recorded commits into the catalogue's upstream rows. */
function updateCatalogue(manifest) {
  if (!existsSync(CATALOGUE)) return;
  const cat = JSON.parse(readFileSync(CATALOGUE, "utf8"));
  let changed = false;
  for (const m of cat.modules) {
    for (const u of m.upstream || []) {
      const e = manifest.entries.find((x) => x.repo === u.repo && x.from === u.path);
      if (e && u.commit !== e.commit) {
        u.commit = e.commit;
        changed = true;
      }
    }
  }
  if (changed) writeFileSync(CATALOGUE, JSON.stringify(cat, null, 2) + "\n");
}

function main() {
  const args = process.argv.slice(2);
  if (!existsSync(MANIFEST)) die(`missing ${path.relative(ROOT, MANIFEST)}`);
  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  if (manifest.schema !== "buhera.vendor/1") die(`unknown manifest schema ${manifest.schema}`);
  const mode = args[0]?.startsWith("--") ? args.shift() : "--check";
  const pick = (ids) =>
    ids.length === 0 || ids.includes("all")
      ? manifest.entries
      : ids.map((id) => manifest.entries.find((e) => e.id === id) || die(`unknown entry ${id}`));

  if (mode === "--list") {
    for (const e of manifest.entries) console.log(`${e.id.padEnd(18)} ${e.repo}:${e.from} → ${e.to}`);
    return;
  }
  if (mode === "--sync") {
    if (args.length === 0) die("--sync needs entry ids or `all`");
    for (const e of pick(args)) {
      const r = sync(manifest, e);
      console.log(`${r.status.padEnd(12)} ${e.id.padEnd(18)} ${r.detail}`);
    }
    writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + "\n");
    updateCatalogue(manifest);
    return;
  }
  if (mode !== "--check") die(`unknown mode ${mode}`);
  let failed = 0;
  for (const e of pick(args)) {
    const r = check(manifest, e);
    if (r.status === "failed") failed++;
    console.log(`${r.status.padEnd(12)} ${e.id.padEnd(18)} ${r.detail}`);
  }
  process.exit(failed ? 1 : 0);
}

main();
