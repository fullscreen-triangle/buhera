// Load the REAL engines, from the copies long-grass vendors (the single TS
// engine store), exactly as the long-grass host injects them.
import { readFile } from "node:fs/promises";
import { loadWasmEngine } from "../src/wasm.ts";
import type { Engines } from "../src/modules/index.ts";

const LG = "../../../long-grass/vendor";

export async function realEngines(): Promise<Required<Engines>> {
  const [smithCompile, smithTown] = await Promise.all([import(`${LG}/agent-smith/src/compile.js`), import(`${LG}/agent-smith/src/town.js`)]);
  const synopsis = await import(`${LG}/synopsis/src/index.ts`);
  const [cfcRun, cfcParse, cfcExamples] = await Promise.all([import(`${LG}/cfc/src/interpreter.js`), import(`${LG}/cfc/src/parser.js`), import(`${LG}/cfc/examples.js`)]);
  const [sth, sthChi] = await Promise.all([import(`${LG}/sthurbert/src/sthurbert/index.ts`), import(`${LG}/sthurbert/src/chi.ts`)]);
  const honjo = await import(`${LG}/honjo/honjo.js`);
  const shapeshifter = await import(`${LG}/shapeshifter/shapeshifter/compiler.js`);
  const ladder = await import(`${LG}/ladder/src/engine.js`);
  const scope = await import(`${LG}/scope-lang/src/index.ts`);
  const [emb, mf] = await Promise.all([import(`${LG}/spectral/src/embedding.js`), import(`${LG}/spectral/src/matched_filter.js`)]);
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
    cfc: { ...cfcRun, ...cfcParse, ...cfcExamples },
    sthurbert: { ...sth, computeCharacter: sthChi.computeCharacter },
    honjo,
    shapeshifter,
    ladder,
    spectral: { ...emb, ...mf },
    scope,
    gateway: { baseUrl: () => "http://127.0.0.1:9", token: () => null },
  } as Required<Engines>;
}
