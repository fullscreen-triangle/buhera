/**
 * Shared types for the zangalewa/desk pipeline.
 *
 * SCoord is the three-axis semantic address described in the MSI paper
 * (S_k, S_t, S_e), each in [0, 1]. RenderResult is what the extractor
 * returns: one or more "leaves" of already-rendered content plus a caption.
 *
 * Only research leaves are defined here for v1. Additional leaf kinds (plot,
 * table, code, etc.) plug in by extending the discriminated union of Leaf.
 */

export interface SCoord {
  /** knowledge specificity: 0 = narrow/specific, 1 = broad/general */
  S_k: number;
  /** temporal entropy: 0 = well-settled, 1 = active frontier */
  S_t: number;
  /** evolution entropy: 0 = simple lookup, 1 = multi-step inference */
  S_e: number;
}

export interface ResearchReference {
  citation: string;
  url: string;
}

export interface ResearchSection {
  heading: string;
  body: string;
}

export interface ResearchParams {
  kind: string;
  title: string;
  tag: string;
  sections: ResearchSection[];
  references: ResearchReference[];
}

export interface ResearchLeaf {
  leaf: "research";
  coord: SCoord;
  params: ResearchParams;
}

/** The single tagged-union of all leaf kinds. Extend as new leaves land. */
export type Leaf = ResearchLeaf;

export interface RenderResult {
  caption: string;
  leaves: Leaf[];
}
