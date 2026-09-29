# spectral — spectral homology and matched-filter motif scanning

| | |
|---|---|
| **Registry id** | `spectral` |
| **Layer** | science |
| **Language** | none |
| **Upstream** | `fullscreen-triangle/gospel` · `vivid-symbolism/src/lib` @ `f4b695b` |
| **Vendored at** | `long-grass/vendor/spectral/src` (byte-exact: `alphabets.js`, `embedding.js`, `fft.js`, `matched_filter.js`) |
| **TS binding** | native — `registry-ts/src/modules/spectral.ts`, bound in `long-grass/src/lib/modules/spectral-module.js` |
| **Rust binding** | none — the engine is JS |

## 1. Purpose

vivid-symbolism is gospel's interactive companion to the shader-based-homology paper. Its numerical core does three things:

1. **Spectral embedding.** A sequence becomes channels (four one-hot channels for DNA; hydropathy, volume and charge for protein), each mean-centred; the first K low-frequency DFT magnitudes per channel form a vector, L2-normalised. Similar sequences have similar low-frequency structure.
2. **Shader-kernel ranking.** A database of embeddings is scanned by dot product — cosine similarity, since the vectors are unit length — and the top K returned.
3. **Matched filtering.** A DNA query is cross-correlated against a target by FFT, normalised per window; background statistics turn scores into z-scores; greedy maximum suppression picks peaks.

These are exactly the operations a synopsis program names — `project … by spectral(coeffs = 8)`, `compare … by shader(cosine)`, `compare … by xcorr(normalised)`, `detect peaks { z; min_distance; min_score }`. **This module is not a synopsis evaluator**: it computes on sequences the caller supplies, with thresholds the caller states. The `synopsis` module checks programs; nothing runs them.

## 2. What is bound

`spectralEmbedding`, `shaderKernelScan`, `prepareTarget`, `matchedFilterScan`, `backgroundStats`, `findPeaks`. Not vendored: `locus.js` (fetches genome regions over the network), `webgl.js` (DOM), and the synopsis IDE copy under `synopsis/` (the `synopsis` module vendors the front end itself).

## 3. Instructions

| Instruction | Effect |
|---|---|
| `"demo"` | rank a small, openly synthetic protein panel against a query |
| `{kind: "embed", sequence, alphabet?: "protein" \| "dna", coeffs?: 1..256 = 8}` | the unit embedding vector |
| `{kind: "homology", query, database: [{name, sequence}], alphabet?, coeffs?, top? = 10}` | cosine ranking |
| `{kind: "motif", query, target, z, min_distance, min_score}` | matched-filter hits; **every threshold must be stated** (as in synopsis, there are no defaults) |

Caps: sequences ≤ 2 000 000 characters; databases ≤ 20 000 entries.

## 4. Output delta

`spectral_embedding` (`{alphabet, coeffs, dim, vector}`), `spectral_ranked` (`{alphabet, coeffs, dim, ranked: [{name, index, cosine}]}`), `spectral_motif` (`{lags, background: {mean, std}, thresholds, hits: [{offset, z, score}]}`). The engine's timing breakdown (`performance.now`) is dropped.

## 5. Residue

0: every operation completes in one act.

## 6. Side effects and hazards

None: pure and deterministic once timings are dropped.

## 7. Conformance

- `registry-ts/test/modules.test.ts` — the identical sequence ranks first with cosine 1 and the unrelated one last; an embedding has unit norm; two copies of a 16-mer planted at offsets 1000 and 2016 in a seeded 3 kb target are exactly the hits at z ≥ 4; a motif request without thresholds is refused.
