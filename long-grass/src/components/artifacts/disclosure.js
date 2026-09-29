/* ============================================================================
 * Disclosure — the single switch between "toggleable" and "flat" rendering.
 *
 * Artifact renderers never hold show/hide state directly; they ask
 * useDisclosure(). Inside <FlatContext.Provider value={true}> every
 * disclosure is permanently open and `flat` is true, so the renderer omits
 * its toggle control. The blank surface renders flat: a page is an inert
 * snapshot, and anything knowable must be visible on it.
 * ========================================================================== */

import { createContext, useContext, useState } from "react";

export const FlatContext = createContext(false);

/**
 * @param {*} initial  initial open value (boolean, or any value for keyed
 *                     disclosures such as "which row is expanded")
 * @returns {{ open: *, setOpen: Function, toggle: Function, flat: boolean }}
 *          In flat mode `open` is `true` regardless of state.
 */
export function useDisclosure(initial = false) {
  const flat = useContext(FlatContext);
  const [open, setOpen] = useState(initial);
  return {
    open: flat ? true : open,
    setOpen,
    toggle: () => setOpen((o) => !o),
    flat,
  };
}
