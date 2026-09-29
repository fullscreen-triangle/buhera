/* ============================================================================
 * SurfaceActions — what a page's contents may ask of the surface.
 *
 * A page is inert: nothing drawn on it can change it. What a page's contents
 * can do is start a NEW step — the same fork-to-the-end every step takes. The
 * surface provides that through this context; anywhere else (the legacy
 * terminal, tutorial cells) it is absent and the affordance is not drawn.
 *
 *   derive(label, produce)   commit a new page whose words are `label` and
 *                            whose envelope is the result of `produce()`
 * ========================================================================== */

import { createContext, useContext } from "react";

export const SurfaceActions = createContext(null);

export function useSurfaceActions() {
  return useContext(SurfaceActions);
}
