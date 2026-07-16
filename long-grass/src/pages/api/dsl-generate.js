// API route: DSL generation.
//
// Compiles an uninformed user's natural-language instructions into a valid
// DSL script (via the FKAC federation + the DSL's real compiler as judge),
// then OPTIONALLY dispatches it to the owning module — the same executor an
// informed user's hand-written script reaches.
//
// Contract:
//   POST /api/dsl-generate
//     body: { dslId: string, instructions: string, execute?: boolean, maxRepairs?: number }
//   -> { ok: true, dslId, code, federation, repairs, attempts, act? }
//   -> { ok: false, dslId, errors, repairs, attempts, code? }
//
//   `act` (present only when execute:true and generation succeeded) is the
//   module's ActResult: { ok, output_delta, residue, completed, error? }.

import { generateDsl } from "@/lib/purpose/dsl-generator";
import { getDsl } from "@/lib/purpose/dsl/validators";
import { dispatch } from "@/lib/modules/registry";

const MAX_INSTRUCTIONS_BYTES = 32 * 1024;
const MAX_REPAIRS_CAP = 6;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { dslId, instructions, execute, maxRepairs } = req.body || {};

  if (typeof dslId !== "string" || !getDsl(dslId)) {
    return res
      .status(400)
      .json({ ok: false, error: `dslId must be one of the registered DSLs; got "${dslId}"` });
  }
  if (typeof instructions !== "string" || instructions.trim().length === 0) {
    return res
      .status(400)
      .json({ ok: false, error: "instructions (non-empty string) is required" });
  }
  if (instructions.length > MAX_INSTRUCTIONS_BYTES) {
    return res.status(413).json({ ok: false, error: "instructions exceed 32 KiB" });
  }

  const repairs =
    Number.isInteger(maxRepairs) && maxRepairs >= 0
      ? Math.min(maxRepairs, MAX_REPAIRS_CAP)
      : 3;

  let gen;
  try {
    gen = await generateDsl({ dslId, instructions, maxRepairs: repairs });
  } catch (err) {
    return res
      .status(500)
      .json({ ok: false, dslId, error: `generation crashed: ${err.message || String(err)}` });
  }

  if (!gen.ok) {
    // A missing provider is an infra problem (503); a code that would not
    // validate after repairs is a generation problem (422).
    const providerStage = (gen.errors || []).some((e) => e.stage === "provider");
    return res.status(providerStage ? 503 : 422).json(gen);
  }

  // Generation succeeded. Dispatch only if asked.
  if (!execute) {
    return res.status(200).json(gen);
  }

  const dsl = getDsl(dslId);
  let act;
  try {
    act = await dispatch(dsl.moduleId, gen.code);
  } catch (err) {
    return res.status(502).json({
      ...gen,
      act: { ok: false, error: `dispatch failed: ${err.message || String(err)}` },
    });
  }

  return res.status(200).json({ ...gen, act });
}
