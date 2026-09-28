/* ============================================================================
 * HFQ module — binds the library adapter (@buhera/registry, specification
 * specifications/specs/hfq.md) to the vendored hegel federated query
 * interpreter @hegel/hfq.
 *
 * The engine runs the paper's pipeline (parse → resolve → check → allocate →
 * execute → emit) against a local fixture world — including Mark Doerr's five
 * real biocatalysis questions (mark_q1–mark_q5) and the eight Chem-DCAT-AP
 * queries (dcat_g1–dcat_g8) — and returns the six-verdict document.
 * ========================================================================== */

import runPlan, { parse, selectWorld } from "@hegel/hfq";
import { PLANS, SECTIONS } from "@hegel/hfq/plans";
import { makeHfqModule } from "@buhera/registry/modules";

export const hfqEngine = { runPlan, parse, selectWorld, PLANS, SECTIONS };
export const hfqModule = makeHfqModule(hfqEngine);
