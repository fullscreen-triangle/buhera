/* ============================================================================
 * purpose-cost.js
 *
 * long-grass's token-cost heuristic for a step's textual content, per buhera
 * specifications §8: cost is caller policy. @buhera/purpose accepts a number;
 * it does not tokenize.
 *
 * v1 heuristic: ceil(content.length / 4). Roughly right for BPE-tokenised
 * English at 3.5-4 chars per token. Users who want an exact count can swap
 * in js-tiktoken here without touching @buhera/purpose or the module adapter.
 * ========================================================================== */

/**
 * Estimate the token cost of a content string.
 *
 * @param {string} content
 * @returns {number} non-negative integer estimated token count
 */
export function estimateCost(content) {
  if (typeof content !== "string" || content.length === 0) return 0;
  return Math.ceil(content.length / 4);
}

/**
 * Estimate the token cost of a whole instruction. Falls back to a small
 * fixed cost when the instruction is a bare object without recognisable
 * text fields (so every dispatched act has cost >= 1).
 *
 * @param {unknown} instruction
 * @returns {number}
 */
export function estimateCostFromInstruction(instruction) {
  if (instruction == null) return 1;
  if (typeof instruction === "string") return Math.max(1, estimateCost(instruction));
  if (typeof instruction === "object") {
    let total = 0;
    for (const v of Object.values(instruction)) {
      if (typeof v === "string") total += estimateCost(v);
      else if (typeof v === "number" || typeof v === "boolean") total += 1;
    }
    return Math.max(1, total);
  }
  return 1;
}
