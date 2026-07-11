/* @sachikonye/sbs — stub.
 *
 * Real runtime lives at ../../hegel/sbs (Rust core + a JS/WebGL2 host that
 * has not yet been packaged as an npm module). This stub is what long-grass
 * consumes in the meantime so its build passes.
 *
 * runSBS() returns an error result explaining what to do. When the real
 * package is linked (e.g. `npm link @sachikonye/sbs` from a locally-built
 * hegel/sbs), this stub is replaced transparently.
 */

const NOT_INSTALLED_MESSAGE =
  "@sachikonye/sbs runtime is not installed. " +
  "SBS scripts cannot execute until the real package is linked from hegel/sbs " +
  "or published to a registry. See long-grass/vendor/sbs-stub for details.";

export function runSBS(_source, _opts) {
  return {
    ok: false,
    error: {
      kind: "runtime-missing",
      message: NOT_INSTALLED_MESSAGE,
    },
    output: [],
    metrics: {},
    diagnostics: [],
  };
}

export const SBS_STUB = true;
