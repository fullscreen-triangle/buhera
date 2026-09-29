// Load the REAL engines, from the copies long-grass vendors (the single TS
// engine store), exactly as the long-grass host injects them.
import { readFile } from "node:fs/promises";
import { loadWasmEngine } from "../src/wasm.ts";
import type { Engines } from "../src/modules/index.ts";

const LG = "../../../long-grass/vendor";

export async function realEngines(): Promise<Required<Engines>> {
  const [smithCompile, smithTown] = await Promise.all([import(`${LG}/agent-smith/src/compile.js`), import(`${LG}/agent-smith/src/town.js`)]);
  const synopsis = await import(`${LG}/synopsis/src/index.ts`);
  const [sbs, hfq, plans, pylon, tcompile, truntime, tconstruct, tcompose, icept] = await Promise.all([
    import(`${LG}/sbs/index.js`),
    import(`${LG}/hfq/src/index.js`),
    import(`${LG}/hfq/src/plans.js`),
    import(`${LG}/pylon/dist/index.js`),
    import(`${LG}/tempus/src/compile.ts`),
    import(`${LG}/tempus/src/runtime.ts`),
    import(`${LG}/tempus/src/construct.ts`),
    import(`${LG}/tempus/src/composition.ts`),
    import(`${LG}/zangalewa-interceptor/client/interceptor-client.ts`),
  ]);
  const wasm = await loadWasmEngine(await readFile(new URL("../wasm/buhera_modules.wasm", import.meta.url)));
  return {
    wasm,
    sbs,
    hfq: { ...hfq, PLANS: plans.PLANS, SECTIONS: plans.SECTIONS },
    pylon,
    tempus: { ...tcompile, ...truntime, ...tconstruct, ...tcompose },
    zangalewa: { Client: icept.Interceptor, baseUrl: "http://127.0.0.1:9" },
    smith: { ...smithCompile, ...smithTown },
    synopsis,
    gateway: { baseUrl: () => "http://127.0.0.1:9", token: () => null },
  } as Required<Engines>;
}
