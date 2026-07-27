/* ============================================================================
 * balance-check — Gap 1 derivation, as a Buhera Solver module.
 *
 * The wire graph asserts three transaminations but nowhere states that they
 * conserve mass or charge — that correctness lives in the partner's pytest
 * suite, invisible to any consumer of the graph. This module makes the
 * correctness a QUERYABLE fact: for each reaction it sums element counts and
 * net charge over substrates vs. products (formulas resolved from ChEBI, since
 * the wire carries none), and reports the residual.
 *
 * Conforms to the Buhera module trait:
 *   { id, describe(), async execute(instruction, actBudget), outputCell() }
 * returning an ActResult { ok, output_delta, residue, completed, error? }.
 *
 * The cofactor (PLP) is regenerated within the ping-pong bi-bi cycle, so it is
 * NOT part of the net stoichiometry and is excluded from the balance sums.
 * ========================================================================== */

import { lookup, monoisotopicMass, hillFormula } from "./chebi-table.mjs";

const CHEBI_PREFIX = "http://purl.obolibrary.org/obo/CHEBI_";

function chebiOf(speciesIri, model) {
  const rec = model.species.get(speciesIri);
  return rec ? rec.chebi : null;
}

function addFormula(acc, formula, sign) {
  for (const [el, n] of Object.entries(formula)) {
    acc[el] = (acc[el] ?? 0) + sign * n;
  }
}

/** Balance one reaction; returns a plain result record (no RDF here). */
function balanceReaction(reaction, model) {
  const unresolved = [];
  const net = {}; // element -> (products - substrates)
  let chargeNet = 0;
  let substrateMass = 0;
  let productMass = 0;

  const side = (iris, sign) => {
    for (const sIri of iris) {
      const acc = chebiOf(sIri, model);
      const rec = acc ? lookup(acc) : null;
      if (!rec) {
        unresolved.push(sIri);
        continue;
      }
      addFormula(net, rec.formula, sign);
      chargeNet += sign * rec.charge;
      const m = monoisotopicMass(rec.formula);
      if (m != null) {
        if (sign > 0) substrateMass += m;
        else productMass += m;
      }
    }
  };

  // substrates count negative, products positive -> net should be all-zero
  side(reaction.substrates, -1);
  side(reaction.products, +1);

  const elementResidual = Object.fromEntries(
    Object.entries(net).filter(([, v]) => v !== 0),
  );
  const massResidual = productMass - substrateMass;

  return {
    reaction: reaction.iri,
    key: reaction.key,
    ec: reaction.ec,
    unresolved,
    massBalanced: unresolved.length === 0 && Math.abs(massResidual) < 1e-6,
    chargeBalanced: unresolved.length === 0 && chargeNet === 0,
    elementResidual, // {} when balanced
    chargeResidual: chargeNet,
    massResidual, // Da, ~0 when balanced
    substrateMass,
    productMass,
    substrateFormula: hillFormulaOfSide(reaction.substrates, model),
    productFormula: hillFormulaOfSide(reaction.products, model),
  };
}

function hillFormulaOfSide(iris, model) {
  const acc = {};
  for (const sIri of iris) {
    const a = chebiOf(sIri, model);
    const rec = a ? lookup(a) : null;
    if (rec) addFormula(acc, rec.formula, +1);
  }
  return hillFormula(acc);
}

/** The Buhera module. Stateless across dispatches; last result cached for outputCell. */
export function makeBalanceCheckModule() {
  let last = null;

  return {
    id: "balance-check",

    describe() {
      return {
        id: "balance-check",
        role: "solver",
        gap: 1,
        summary:
          "Derives mass- and charge-balance of each reaction from ChEBI formulas and reports the residual as queryable facts.",
      };
    },

    /**
     * @param instruction { kind:"balance", model } — the ingest model
     * @param actBudget    number of reactions to attempt this act (default all)
     */
    async execute(instruction, actBudget = 1) {
      if (!instruction || instruction.kind !== "balance" || !instruction.model) {
        return {
          ok: false,
          output_delta: null,
          residue: 0,
          completed: false,
          error: "balance-check expects { kind:'balance', model }",
        };
      }
      const { model } = instruction;
      const budget =
        actBudget >= model.reactions.length || actBudget <= 0
          ? model.reactions.length
          : actBudget;
      const results = model.reactions
        .slice(0, budget)
        .map((r) => balanceReaction(r, model));

      const unresolvedCount = results.reduce(
        (n, r) => n + r.unresolved.length,
        0,
      );
      const allBalanced = results.every(
        (r) => r.massBalanced && r.chargeBalanced,
      );

      last = { results, allBalanced };
      return {
        ok: true,
        output_delta: { kind: "balance_report", results, allBalanced },
        // residue = work left undone: species we could not resolve + reactions skipped
        residue: unresolvedCount + (model.reactions.length - budget),
        completed: budget >= model.reactions.length,
      };
    },

    outputCell() {
      return last;
    },
  };
}
