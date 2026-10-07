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
 *   write(text)              commit `text` as if it had been typed
 *   draft(text)              put `text` under the caret, uncommitted
 *
 * useStep() wraps derive for the common case: run one module instruction and
 * show its result as the next page. It returns null outside the surface.
 * ========================================================================== */

import { createContext, useCallback, useContext } from "react";
import { dispatch } from "@/lib/modules/registry";

export const SurfaceActions = createContext(null);

export function useSurfaceActions() {
  return useContext(SurfaceActions);
}

export function useStep() {
  const actions = useContext(SurfaceActions);
  const step = useCallback(
    (label, moduleId, instruction) =>
      actions?.derive(label, async () => {
        const r = await dispatch(moduleId, instruction);
        return r?.output_delta ? { kind: "artifact", result: r.output_delta } : { kind: "text", lines: [`(${moduleId}: no output)`] };
      }),
    [actions]
  );
  return actions ? step : null;
}
