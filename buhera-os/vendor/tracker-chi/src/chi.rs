//! The character invariant χ (T1) — a repo's conserved sense/goal.
//!
//! Theory (contact-graph foundation + split-attention T1): a repo is a finite
//! weighted graph; its character invariant is the *minimum cut-residual* — the least
//! total weight of splitting it into pieces, in the currency of its own separations.
//! The invariant is **positive** (T0 floor β > 0), **conserved** under relabelling
//! (a weighted-graph invariant), and **non-local** (realised by a multi-block
//! partition, never a single vertex).
//!
//! ## Realisation decisions (made deliberately; see design doc §3, §9 Q1)
//!
//! * **Vertices = files/sections**, not individual symbols. We aggregate the
//!   `purpose` index's symbols into their containing file. This keeps |V| in the
//!   hundreds (Stoer–Wagner is O(V³)), and is the *faithful* coarsening: a repo's
//!   sense splits along subsystem/file boundaries, not between two functions in one
//!   file.
//! * **Edges = containment + reference proximity.** Two files are joined with weight
//!   = (shared directory depth) + (name/snippet cross-references). Concretely: every
//!   pair of files under a common directory contributes `β` per shared path
//!   component, and each time one file's symbol name appears in another file's
//!   snippets contributes `β`. All weights are floored at `β = BETA`.
//! * **χ = global minimum cut** (min over bipartitions of crossing weight), computed
//!   exactly by Stoer–Wagner. For r=2 this *is* the min cut-residual; it is exactly
//!   the quantity the paper's two-triangle witness uses, and it is non-local by
//!   construction (it returns a set of vertices, not one).
//!
//! The floor β guarantees χ ≥ β > 0 whenever the graph is connected with ≥ 2 blocks.

use crate::purpose::Index;
use std::collections::BTreeMap;

/// The weight floor β > 0 (T0). Every present separation costs at least this.
pub const BETA: f64 = 1.0;

/// The computed character invariant for a repo, plus the salient surface that
/// realises it. χ itself is a scalar; the salient blocks are the human-readable
/// "sense" (reported *by search*, this struct only names where to look).
#[derive(Debug, Clone)]
pub struct Character {
    /// χ — the minimum cut-residual **of the largest connected component**.
    /// Positive and conserved (T1). See `fragments` for why the component matters.
    pub chi: f64,
    /// Number of file/section blocks (graph vertices) across the whole repo.
    pub blocks: usize,
    /// Number of connected components. The floor theorem (T0) guarantees χ > 0 only
    /// for a *connected* graph (the Presence axiom); a repo's index is naturally
    /// fragmented (docs vs. code), so χ is the invariant of the principal body — the
    /// largest component — and this counts the islands as a structural fact.
    pub fragments: usize,
    /// Size (in blocks) of the largest connected component χ was computed over.
    pub core_blocks: usize,
    /// The smaller side of the minimum cut within the core — the block set most
    /// cheaply severed. Non-local: a *region*, never guaranteed to be a singleton.
    pub cut_side: Vec<String>,
    /// Highest-degree blocks — the load-bearing files of the repo's structure.
    /// This is the salient surface a `sense` report narrates.
    pub salient: Vec<(String, f64)>,
}

/// Compute χ and its salient surface from a repo's `purpose` self-graph.
///
/// χ is the minimum cut-residual of the **largest connected component** (the repo's
/// principal body of work). This is the theory-faithful choice: the floor theorem
/// guarantees χ ≥ β only under connectedness (the Presence axiom), and a repo's raw
/// index graph is naturally disconnected. We report the component count as
/// `fragments` so fragmentation is surfaced, never hidden.
pub fn compute(index: &Index) -> Character {
    let graph = build_graph(index);
    let fragments = graph.components();
    let salient = graph.top_by_degree(8);
    let core = graph.largest_component_subgraph();
    let mut character = core.character();
    // Overwrite whole-repo-level facts the sub-graph cannot know.
    character.blocks = graph.labels.len();
    character.fragments = fragments.len();
    character.core_blocks = core.labels.len();
    character.salient = salient;
    character
}

/// A weighted, undirected graph over file/section blocks.
struct BlockGraph {
    labels: Vec<String>,
    /// Symmetric adjacency; `w[i][j]` is the total separation cost between block i,j.
    w: Vec<Vec<f64>>,
}

impl BlockGraph {
    /// Character of THIS graph, assumed connected (caller passes a component).
    fn character(&self) -> Character {
        let n = self.labels.len();
        if n < 2 {
            // A component with fewer than two blocks has no nontrivial partition; by
            // the floor theorem its sense is still positive but undefined as a cut —
            // report β as the floor and the whole thing as salient.
            return Character {
                chi: BETA,
                blocks: n,
                fragments: 1,
                core_blocks: n,
                cut_side: self.labels.clone(),
                salient: self.top_by_degree(8),
            };
        }
        let (chi, side) = self.stoer_wagner_min_cut();
        let cut_side = side.into_iter().map(|i| self.labels[i].clone()).collect();
        Character {
            chi,
            blocks: n,
            fragments: 1,
            core_blocks: n,
            cut_side,
            salient: self.top_by_degree(8),
        }
    }

    /// Connected components as lists of vertex indices, largest first.
    fn components(&self) -> Vec<Vec<usize>> {
        let n = self.labels.len();
        let mut seen = vec![false; n];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut stack = vec![start];
            seen[start] = true;
            let mut comp = Vec::new();
            while let Some(v) = stack.pop() {
                comp.push(v);
                for u in 0..n {
                    if !seen[u] && self.w[v][u] > 0.0 {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
            comps.push(comp);
        }
        comps.sort_by(|a, b| b.len().cmp(&a.len()));
        comps
    }

    /// The induced subgraph on the largest connected component.
    fn largest_component_subgraph(&self) -> BlockGraph {
        let comps = self.components();
        let core = comps.first().cloned().unwrap_or_default();
        let labels: Vec<String> = core.iter().map(|&i| self.labels[i].clone()).collect();
        let k = core.len();
        let mut w = vec![vec![0.0f64; k]; k];
        for (ni, &oi) in core.iter().enumerate() {
            for (nj, &oj) in core.iter().enumerate() {
                w[ni][nj] = self.w[oi][oj];
            }
        }
        BlockGraph { labels, w }
    }

    fn top_by_degree(&self, k: usize) -> Vec<(String, f64)> {
        let mut ds: Vec<(String, f64)> = self
            .w
            .iter()
            .enumerate()
            .map(|(i, row)| (self.labels[i].clone(), row.iter().sum::<f64>()))
            .collect();
        ds.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0)) // deterministic tie-break
        });
        ds.truncate(k);
        ds
    }

    /// Stoer–Wagner global minimum cut. Deterministic; O(V³).
    /// Returns (cut weight, smaller side as vertex indices into the ORIGINAL labels).
    fn stoer_wagner_min_cut(&self) -> (f64, Vec<usize>) {
        let n = self.labels.len();
        let mut w = self.w.clone();
        // `groups[v]` = original vertices merged into current vertex v.
        let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut alive: Vec<usize> = (0..n).collect();
        let mut best_cut = f64::INFINITY;
        let mut best_side: Vec<usize> = Vec::new();

        while alive.len() > 1 {
            // Minimum-cut-phase (maximum adjacency search).
            let m = alive.len();
            let mut added = vec![false; m];
            let mut weights = vec![0.0f64; m];
            let mut order: Vec<usize> = Vec::with_capacity(m);
            for _ in 0..m {
                // pick the most tightly connected not-yet-added vertex
                let mut sel = usize::MAX;
                let mut best = f64::NEG_INFINITY;
                for i in 0..m {
                    if !added[i] && weights[i] > best {
                        best = weights[i];
                        sel = i;
                    }
                }
                added[sel] = true;
                order.push(sel);
                for i in 0..m {
                    if !added[i] {
                        weights[i] += w[alive[sel]][alive[i]];
                    }
                }
            }
            // Last two added: `t` (last), `s` (second last).
            let t_local = order[m - 1];
            let s_local = order[m - 2];
            let cut_of_phase = weights[t_local]; // weight into the last-added vertex
            let t = alive[t_local];
            let s = alive[s_local];

            if cut_of_phase < best_cut {
                best_cut = cut_of_phase;
                best_side = groups[t].clone();
            }

            // Merge t into s.
            let t_group = std::mem::take(&mut groups[t]);
            groups[s].extend(t_group);
            for &x in &alive {
                if x != s && x != t {
                    w[s][x] += w[t][x];
                    w[x][s] += w[x][t];
                }
            }
            alive.retain(|&x| x != t);
        }

        // Report the smaller side for a stable, non-local witness.
        let side = if best_side.len() * 2 <= self.labels.len() {
            best_side
        } else {
            let in_side: std::collections::HashSet<usize> = best_side.into_iter().collect();
            (0..self.labels.len())
                .filter(|i| !in_side.contains(i))
                .collect()
        };
        let mut side = side;
        side.sort_unstable();
        (best_cut, side)
    }
}

/// Build the block graph from the index: aggregate symbols by file, then join files
/// by containment (shared path depth) and reference proximity (name mentions).
fn build_graph(index: &Index) -> BlockGraph {
    // 1. Collect files (blocks) in first-seen order → stable labels.
    let mut label_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut labels: Vec<String> = Vec::new();
    // Per-file: the set of symbol names it defines, and its snippet text (lowercased).
    let mut names_in: Vec<Vec<String>> = Vec::new();
    let mut snippets_in: Vec<String> = Vec::new();

    for s in &index.symbols {
        let id = *label_of.entry(s.file.clone()).or_insert_with(|| {
            labels.push(s.file.clone());
            names_in.push(Vec::new());
            snippets_in.push(String::new());
            labels.len() - 1
        });
        names_in[id].push(s.name.clone());
        snippets_in[id].push_str(&s.snippet.to_lowercase());
        snippets_in[id].push('\n');
    }

    let n = labels.len();
    let mut w = vec![vec![0.0f64; n]; n];

    // 2. Containment: files sharing directory prefix components are separated more
    //    cheaply from the outside the deeper they nest together → weight per shared
    //    leading path component.
    let comps: Vec<Vec<&str>> = labels
        .iter()
        .map(|f| f.split(['/', '\\']).collect())
        .collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let shared = comps[i]
                .iter()
                .zip(comps[j].iter())
                .take_while(|(a, b)| a == b)
                .count();
            if shared > 0 {
                add_edge(&mut w, i, j, BETA * shared as f64);
            }
        }
    }

    // 3. Reference proximity: file i's defined name appearing in file j's snippets.
    //    Restrict to distinctive names (len ≥ 4) to avoid `i`, `id`, `x` noise.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut refs = 0usize;
            for name in &names_in[i] {
                if name.len() >= 4 && snippets_in[j].contains(&name.to_lowercase()) {
                    refs += 1;
                }
            }
            if refs > 0 {
                add_edge(&mut w, i, j, BETA * refs as f64);
            }
        }
    }

    BlockGraph { labels, w }
}

fn add_edge(w: &mut [Vec<f64>], i: usize, j: usize, delta: f64) {
    w[i][j] += delta;
    w[j][i] += delta;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::purpose::Symbol;

    fn sym(name: &str, file: &str, snippet: &str) -> Symbol {
        Symbol {
            name: name.into(),
            kind: "def".into(),
            file: file.into(),
            line: 1,
            snippet: snippet.into(),
        }
    }

    /// The paper's non-locality witness, in file form: two dense clusters joined by a
    /// single cheap link. χ must be the cross-cluster cut, and the cut side must be a
    /// *region* (multiple blocks), never a singleton.
    #[test]
    fn chi_is_positive_and_non_local() {
        // cluster A: three files under a/ that reference each other
        // cluster B: three files under b/ that reference each other
        // one weak link: a/one mentions bravo1 (in b/one)
        let symbols = vec![
            sym("alpha1", "a/one.rs", "alpha2 alpha3"),
            sym("alpha2", "a/two.rs", "alpha1 alpha3"),
            sym("alpha3", "a/three.rs", "alpha1 alpha2"),
            sym("bravo1", "b/one.rs", "bravo2 bravo3"),
            sym("bravo2", "b/two.rs", "bravo1 bravo3"),
            sym("bravo3", "b/three.rs", "bravo1 bravo2"),
            // the single weak inter-cluster reference:
            sym("bridge", "a/one.rs", "bravo1"),
        ];
        let idx = Index {
            root: ".".into(),
            symbols,
        };
        let c = compute(&idx);
        assert!(c.chi >= BETA, "chi must be >= floor, got {}", c.chi);
        assert!(c.chi.is_finite());
        // non-local: the cheap cut separates a *cluster*, not a lone file.
        assert!(
            c.cut_side.len() >= 2 || c.blocks <= 2,
            "cut side should be a region: {:?}",
            c.cut_side
        );
    }

    /// I1 (conserved identity): χ is invariant under relabelling the vertices.
    /// Here we relabel by shuffling the symbol/file order; χ must not move.
    #[test]
    fn chi_invariant_under_relabelling() {
        let base = vec![
            sym("aaaa", "x/p.rs", "bbbb"),
            sym("bbbb", "x/q.rs", "aaaa cccc"),
            sym("cccc", "y/r.rs", "bbbb"),
        ];
        let mut shuffled = base.clone();
        shuffled.reverse(); // a relabelling: same graph, different vertex order
        let c1 = compute(&Index {
            root: ".".into(),
            symbols: base,
        });
        let c2 = compute(&Index {
            root: ".".into(),
            symbols: shuffled,
        });
        assert!(
            (c1.chi - c2.chi).abs() < 1e-12,
            "chi changed under relabelling: {} vs {}",
            c1.chi,
            c2.chi
        );
    }
}
