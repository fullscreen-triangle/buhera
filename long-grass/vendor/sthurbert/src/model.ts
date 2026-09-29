/** Shared analysed-federation model consumed by both input paths (st-Hurbert & AI). */

import type { Character, Symbol } from "./chi";
import type { RepoSnapshot } from "./github";
import type { ImageLineage } from "./lineage";

/**
 * Where an analysed repo came from — the two data sources the cockpit supports.
 *
 *  - `github`: fetched over api.github.com from the browser (needs a token, is
 *    rate-limited, sees only what the API exposes — never uncommitted work).
 *  - `local`: read from the filesystem by the downloaded local engine (§7.6).
 *    No token, sees the working tree (including uncommitted changes), and is the
 *    only origin on which the image-knowability verbs (build/exercise/judge) can
 *    ever run — a browser sandbox has no filesystem, subprocess, or Docker.
 *
 * The same visualisations render either origin; only identity and file-link
 * derivation differ, which is what these helpers below localise.
 */
export type Origin =
  | { kind: "github"; owner: string; name: string; defaultBranch: string }
  | { kind: "local"; path: string; name: string; commit?: string };

/** One fully-analysed repo. */
export interface AnalysedRepo {
  /** Where this repo came from — GitHub fetch or local-engine filesystem read. */
  origin: Origin;
  snapshot: RepoSnapshot;
  symbols: Symbol[];
  character: Character;
  /**
   * The repo's image lineage — its runtime-health-over-time record (the second
   * lineage beside χ's sense-over-time). Optional: absent until any image version
   * has been built/exercised/judged. The read verbs report its absence honestly
   * rather than fabricating a regime map.
   */
  lineage?: ImageLineage;
}

/** Origin for a GitHub-fetched snapshot (the browser path). */
export function githubOrigin(snapshot: RepoSnapshot): Origin {
  return {
    kind: "github",
    owner: snapshot.ref.owner,
    name: snapshot.ref.name,
    defaultBranch: snapshot.defaultBranch,
  };
}

/**
 * Origin for a local-engine filesystem read (the §7.6 downloaded-engine path).
 * The analogue of {@link githubOrigin}: identity is the on-disk path, `name` is
 * the trailing segment the engine reports, `commit` is the working-tree HEAD if
 * the engine resolved one.
 */
export function localOrigin(eng: { path: string; name: string; commit?: string }): Origin {
  return { kind: "local", path: eng.path, name: eng.name, commit: eng.commit };
}

/** The analysed group of repos the query paths run against. */
export interface Federation {
  repos: AnalysedRepo[];
}

/** Canonical identity: `owner/name` for GitHub, the filesystem path for local. */
export function repoName(r: AnalysedRepo): string {
  return originName(r.origin);
}

export function originName(o: Origin): string {
  return o.kind === "github" ? `${o.owner}/${o.name}` : o.path;
}

/** Short label for cramped UI (graph nodes): the last path/name segment. */
export function repoShortName(r: AnalysedRepo): string {
  return r.origin.name;
}

export function findRepo(fed: Federation, target: string): AnalysedRepo | undefined {
  return fed.repos.find((r) => {
    const o = r.origin;
    if (originName(o) === target || o.name === target) return true;
    // local repos also match on a trailing path segment or a normalised path
    if (o.kind === "local") {
      const norm = (s: string) => s.replace(/\\/g, "/").replace(/\/+$/, "");
      return norm(o.path) === norm(target) || norm(o.path).endsWith("/" + norm(target));
    }
    return false;
  });
}

/**
 * A clickable location for a file (optionally at a line), or null when the origin
 * has none. GitHub → a blob URL; local → null for now (the local engine will later
 * supply an editor/`file://` link — never a fabricated github.com URL for a repo
 * that does not live there).
 */
export function fileUrl(r: AnalysedRepo, file: string, line?: number): string | null {
  const o = r.origin;
  if (o.kind !== "github") return null;
  const anchor = line ? `#L${line}` : "";
  return `https://github.com/${o.owner}/${o.name}/blob/${o.defaultBranch}/${file}${anchor}`;
}

/** @deprecated use {@link fileUrl}; retained so callers migrate without breakage. */
export function githubFileUrl(r: AnalysedRepo, file: string, line?: number): string | null {
  return fileUrl(r, file, line);
}
