/* ============================================================================
 * Vendored ChEBI reference — public chemistry, resolved by accession.
 *
 * The wire graph carries only ChEBI IRIs (skos:exactMatch). It does NOT carry
 * formulas or charges. To DERIVE mass/charge balance, Buhera must resolve each
 * accession to its elemental composition. These values are public ChEBI facts
 * (Formula + Charge annotations), vendored here so the derivation is:
 *
 *   - reproducible offline (no live ChEBI call in the hot path);
 *   - decoupled from the partner pipeline (we never read nfdi4cat's reference.py).
 *
 * Each entry: neutral-formula element counts + net charge of the species as
 * modelled by ChEBI at physiological pH. Keyed by bare ChEBI accession.
 * Source: ChEBI (https://www.ebi.ac.uk/chebi/), Formula & Charge annotations.
 * ========================================================================== */

export const CHEBI = {
  // L-alanine  C3H7NO2, neutral zwitterion
  "16977": { name: "L-alanine", formula: { C: 3, H: 7, N: 1, O: 2 }, charge: 0 },
  // pyruvate  C3H3O3(-1)
  "15361": { name: "pyruvate", formula: { C: 3, H: 3, O: 3 }, charge: -1 },
  // L-aspartate  C4H6NO4(-1)
  "17053": { name: "L-aspartate", formula: { C: 4, H: 6, N: 1, O: 4 }, charge: -1 },
  // oxaloacetate  C4H2O5(-2)
  "16452": { name: "oxaloacetate", formula: { C: 4, H: 2, O: 5 }, charge: -2 },
  // L-cysteine  C3H7NO2S, neutral
  "17561": { name: "L-cysteine", formula: { C: 3, H: 7, N: 1, O: 2, S: 1 }, charge: 0 },
  // 2-oxo-3-sulfanylpropanoate  C3H3O3S(-1)
  "16208": { name: "2-oxo-3-sulfanylpropanoate", formula: { C: 3, H: 3, O: 3, S: 1 }, charge: -1 },
  // L-glutamate  C5H8NO4(-1)
  "16015": { name: "L-glutamate", formula: { C: 5, H: 8, N: 1, O: 4 }, charge: -1 },
  // 2-oxoglutarate  C5H4O5(-2)
  "16810": { name: "2-oxoglutarate", formula: { C: 5, H: 4, O: 5 }, charge: -2 },
  // pyridoxal 5'-phosphate  C8H8NO6P(-2)  (cofactor; regenerated, not net-consumed)
  "18405": { name: "pyridoxal 5'-phosphate", formula: { C: 8, H: 8, N: 1, O: 6, P: 1 }, charge: -2 },
};

/** Monoisotopic masses (Da) of the elements that appear in this graph. */
export const MONOISOTOPIC = {
  C: 12.0,
  H: 1.0078250319,
  N: 14.0030740052,
  O: 15.9949146221,
  P: 30.97376151,
  S: 31.97207069,
};

/** Resolve a ChEBI accession to its vendored record, or null if unknown. */
export function lookup(accession) {
  return CHEBI[accession] ?? null;
}

/** Monoisotopic mass of an element-count map. */
export function monoisotopicMass(formula) {
  let m = 0;
  for (const [el, n] of Object.entries(formula)) {
    if (!(el in MONOISOTOPIC)) return null; // unknown element -> cannot mass
    m += MONOISOTOPIC[el] * n;
  }
  return m;
}

/** Render an element-count map as a Hill-order formula string (C, H, then A–Z). */
export function hillFormula(formula) {
  const parts = [];
  const emit = (el) => {
    if (formula[el]) parts.push(el + (formula[el] > 1 ? formula[el] : ""));
  };
  emit("C");
  emit("H");
  for (const el of Object.keys(formula).sort()) {
    if (el !== "C" && el !== "H") emit(el);
  }
  return parts.join("");
}
