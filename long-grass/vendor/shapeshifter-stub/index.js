/* @lavoisier/shapeshifter — stub.
 *
 * Real runtime lives in the lavoisier repo and hasn't been packaged yet.
 */

const NOT_INSTALLED = "@lavoisier/shapeshifter runtime is not installed on this deployment.";

export function compileStage(_source) {
  return {
    ok: false,
    error: { kind: "runtime-missing", message: NOT_INSTALLED },
    diagnostics: [],
    ir: null,
  };
}

export async function executeStage(_ir, _input) {
  return {
    ok: false,
    error: { kind: "runtime-missing", message: NOT_INSTALLED },
    output: null,
    metrics: {},
  };
}

export const SHAPESHIFTER_STUB = true;
