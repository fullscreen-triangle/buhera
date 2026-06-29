/* ============================================================================
 * Echo Module
 *
 * The simplest possible Buhera module: it returns whatever instruction it
 * receives. Exists so the federation has at least two members at v1, which
 * lets us demonstrate that `dispatch(...)` routes by module id and that the
 * audit log records each call.
 *
 * Use from a turbulance script:
 *
 *     item r = dispatch("echo", "hello world")
 *     print(r.output_delta.value)
 *
 * Once real second/third modules land (purpose, geolocate, etc.) this can
 * be deleted or kept as a smoke-test target.
 * ========================================================================== */

export const echoModule = {
  id: "echo",

  describe() {
    return {
      id: "echo",
      description: "echoes its instruction; smoke-test module",
      instructions: [
        'dispatch("echo", <any string or value>)',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    return {
      ok: true,
      output_delta: {
        kind: "echo",
        value: instruction,
      },
      residue: 1,
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "echo_cell" };
  },
};
