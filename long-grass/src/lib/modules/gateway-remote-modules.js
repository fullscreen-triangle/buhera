/* ============================================================================
 * Rust-only modules reached on buhera-gateway (specification 07 §2):
 * currently sbs-core. The act runs in the signed-in account's federation on
 * the gateway; the result comes back verbatim with `executed_on`.
 * ========================================================================== */

import { GATEWAY_MODULES, makeRemoteModule } from "@buhera/registry/modules";
import { gatewayTransport } from "@/lib/modules/gateway-module";

export const gatewayRemoteModules = GATEWAY_MODULES.map((d) => makeRemoteModule(d, gatewayTransport));
