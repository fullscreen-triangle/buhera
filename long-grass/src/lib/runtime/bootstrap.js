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
