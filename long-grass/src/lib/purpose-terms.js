/* ============================================================================
 * purpose-terms.js
 *
 * long-grass's implementation of the term map τ that turns a step's textual
 * content into a Set<Term> for the purpose library. See buhera specifications
 * §8 and the reconciliation doc: τ is caller policy, not @buhera/purpose's.
 *
 * v1 policy:
 *   • lowercase
 *   • split on non-alphanumeric
 *   • keep tokens with length >= 3
 *   • drop a small English stopword list
 *   • dedupe into a Set
 *
 * This is deliberately trivial. Callers who want richer term extraction can
 * swap in NER, embedding-derived tags, or an LLM-driven τ without touching
 * @buhera/purpose or the module adapter.
 * ========================================================================== */

const STOPWORDS = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
  "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new",
  "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put",
  "say", "she", "too", "use", "with", "have", "this", "that", "from", "they",
  "will", "would", "there", "their", "what", "about", "which", "when", "make",
  "like", "time", "just", "know", "take", "into", "your", "some", "them",
  "than", "then", "look", "only", "come", "over", "think", "also", "back",
  "after", "work", "first", "well", "even", "want", "because", "these",
  "give", "most", "very", "still", "should", "could", "does", "here", "each",
]);

/**
 * Extract a Set<Term> from arbitrary content text.
 *
 * @param {string} content
 * @returns {Set<string>} set of lowercase, alphanumeric terms of length >= 3
 */
export function extractTerms(content) {
  if (typeof content !== "string" || content.length === 0) {
    return new Set();
  }
  const out = new Set();
  const tokens = content.toLowerCase().split(/[^a-z0-9]+/);
  for (const t of tokens) {
    if (t.length < 3) continue;
    if (STOPWORDS.has(t)) continue;
    out.add(t);
  }
  return out;
}

/**
 * Convenience: extract terms from a structured instruction. If the instruction
 * is a string, treat it as the content directly. If it's an object, gather
 * text from a small set of predictable fields.
 *
 * @param {unknown} instruction
 * @returns {Set<string>}
 */
export function extractTermsFromInstruction(instruction) {
  if (instruction == null) return new Set();
  if (typeof instruction === "string") return extractTerms(instruction);
  if (typeof instruction === "object") {
    const fields = [
      instruction.description,
      instruction.query,
      instruction.utterance,
      instruction.source,
      instruction.text,
      instruction.claim,
    ];
    const parts = fields.filter((f) => typeof f === "string");
    return extractTerms(parts.join(" "));
  }
  return new Set();
}
