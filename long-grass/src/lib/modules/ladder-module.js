// ladder — levinthal's catalytic-ladder contact-graph engine, bound to the
// vendored enzymes/web engine.js (vendor/ladder, byte-exact). Specification:
// specs/ladder.md. Every verdict is the engine's Machine.runVerdict.
//
// Its derived powers are what an HFQ plan's `ladder over … power P` step
// otherwise takes as bare declared numbers.
import * as ladderEngine from "@levinthal/ladder";
import { makeLadderModule } from "@buhera/registry/modules";

export { ladderEngine };
export const ladderModule = makeLadderModule(ladderEngine);
