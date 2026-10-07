/* ============================================================================
 * Runtime bootstrap — the shared federation registration.
 *
 * Registers every module and wires the post-dispatch hooks (purpose-carry
 * feeder, desk feeder). Called at page mount by both the main terminal and
 * the tutorial pages so both share one federation instance.
 *
 * Idempotent: calling bootstrapFederation() twice does nothing on the second
 * call. The module registry is a process-lifetime singleton — once
 * populated, it stays populated for the browser session.
 * ========================================================================== */

import { register, onDispatch } from "@/lib/modules/registry";
import { vaheraModule } from "@/lib/modules/vahera-module";
import { echoModule } from "@/lib/modules/echo-module";
import { lavoisierModule } from "@/lib/modules/lavoisier-module";
import { purposeModule } from "@/lib/modules/purpose-module";
import { purposeCliModule } from "@/lib/modules/purpose-cli-module";
import { zangalewaModule } from "@/lib/modules/zangalewa-module";
import { graffitiModule } from "@/lib/modules/graffiti-module";
import { purposeCarryModule, getSession as getPurposeSession } from "@/lib/modules/purpose-carry-module";
import { shapeshifterModule } from "@/lib/modules/shapeshifter-module";
import { sbsModule } from "@/lib/modules/sbs-module";
import { scopeModule } from "@/lib/modules/scope-module";
import { catalystRegistryModule } from "@/lib/modules/catalyst-registry-module";
import { computeModule } from "@/lib/modules/compute-module";
import { deskModule, observeAct as deskObserveAct } from "@/lib/modules/desk-module";
import { dslWriterModule } from "@/lib/modules/dsl-writer-module";
import { srnModule } from "@/lib/modules/srn-module";
import { smithModule } from "@/lib/modules/smith-module";
import { synopsisModule } from "@/lib/modules/synopsis-module";
import { cfcModule } from "@/lib/modules/cfc-module";
import { sthurbertModule } from "@/lib/modules/sthurbert-module";
import { honjoModule } from "@/lib/modules/honjo-module";
import { spectralModule } from "@/lib/modules/spectral-module";
import { ckgModule } from "@/lib/modules/ckg-module";
import { cytochromeModule } from "@/lib/modules/cytochrome-module";
import { gatewayModule } from "@/lib/modules/gateway-module";
import { triangleModule } from "@/lib/modules/triangle-module";
import { spraypaintModule } from "@/lib/modules/spraypaint-module";
import { hfqModule } from "@/lib/modules/hfq-module";
import { ladderModule } from "@/lib/modules/ladder-module";
import { interceptorModule } from "@/lib/modules/interceptor-module";
import { systemModules } from "@/lib/modules/system-modules";
import { visModule } from "@/lib/modules/vis-module";
import { surfaceModules } from "@/lib/modules/surface-modules";
import { mailModule } from "@/lib/modules/mail-module";
import { latticeModule } from "@/lib/modules/lattice-module";
import { planningModule } from "@/lib/modules/planning-module";
// Library federation (specifications/registry/catalogue.json): adapters from
// @buhera/registry bound to vendored engines, plus the Rust modules via wasm.
import { pylonModule } from "@/lib/modules/pylon-module";
import { tempusModule } from "@/lib/modules/tempus-module";
import { zangalewaDslModule } from "@/lib/modules/zangalewa-dsl-module";
import { rustWasmModules } from "@/lib/modules/rust-wasm-modules";
import { gatewayRemoteModules } from "@/lib/modules/gateway-remote-modules";
import { extractTermsFromInstruction } from "@/lib/purpose-terms";
import { estimateCostFromInstruction } from "@/lib/purpose-cost";

let _bootstrapped = false;
let _hookCleanup = null;

/**
 * Register every module and wire the purpose-carry audit-log feeder.
 * Returns a cleanup function that removes the feeder hook (module
 * registrations stay — they're process-lifetime).
 */
export function bootstrapFederation() {
  if (_bootstrapped) return _hookCleanup || (() => {});
  _bootstrapped = true;

  register(vaheraModule);
  register(echoModule);
  register(lavoisierModule);
  register(purposeModule);
  register(purposeCliModule);
  register(zangalewaModule);
  register(graffitiModule);
  register(purposeCarryModule);
  register(shapeshifterModule);
  register(sbsModule);
  register(scopeModule);
  register(catalystRegistryModule);
  register(computeModule);
  register(deskModule);
  register(dslWriterModule);
  register(srnModule);
  register(smithModule);
  register(synopsisModule);
  register(cfcModule);
  register(sthurbertModule);
  register(honjoModule);
  register(spectralModule);
  register(ckgModule);
  register(cytochromeModule);
  register(gatewayModule);
  register(triangleModule);
  register(spraypaintModule);
  register(hfqModule);
  register(ladderModule);
  register(interceptorModule);
  register(pylonModule);
  register(tempusModule);
  register(zangalewaDslModule);
  for (const m of rustWasmModules) register(m);
  for (const m of gatewayRemoteModules) register(m);
  for (const m of systemModules) register(m);
  register(visModule);
  for (const m of surfaceModules) register(m);
  register(mailModule);
  register(latticeModule);
  register(planningModule);

  const session = getPurposeSession();
  const unhook = onDispatch((entry) => {
    if (entry.module_id === "purpose-carry") return;
    try {
      const terms = extractTermsFromInstruction(entry.instruction);
      if (terms.size === 0) return;
      const cost = estimateCostFromInstruction(entry.instruction);
      session.addStep({
        id: `act-${entry.act_id}`,
        terms,
        cost,
        timestamp: Date.parse(entry.timestamp) || Date.now(),
        payload: {
          module_id: entry.module_id,
          act_id: entry.act_id,
        },
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("purpose feeder failed for act", entry.act_id, err);
    }
  });

  // Desk observer: every dispatch gets a chance to nudge the desk's
  // standing intent (no-op until one is tagged).
  const unhookDesk = onDispatch((entry) => {
    try {
      deskObserveAct(entry);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("desk observer failed for act", entry.act_id, err);
    }
  });

  _hookCleanup = () => {
    try { unhook(); } catch { /* noop */ }
    try { unhookDesk(); } catch { /* noop */ }
    _hookCleanup = null;
    _bootstrapped = false;
  };
  return _hookCleanup;
}
