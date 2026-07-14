/* scope-lang — stub.
 *
 * Real runtime lives elsewhere and hasn't been packaged yet. This stub is
 * what long-grass consumes in the meantime so its build passes. Replace by
 * pointing package.json at the real package when it exists.
 */

const NOT_INSTALLED = "scope-lang runtime is not installed on this deployment.";

export function createSession(_opts) {
  return {
    run: async () => ({
      ok: false,
      error: { kind: "runtime-missing", message: NOT_INSTALLED },
      output: [],
      diagnostics: [],
    }),
    reset: () => {},
    stats: () => ({ stub: true }),
  };
}

export const SCOPE_LANG_STUB = true;
