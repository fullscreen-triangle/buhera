/**
 * Typed errors (contract §5.8: errors are typed, never thrown from the core).
 * These are the failure variants of a submitted expression's Yield.
 */
import type { Coord } from "./coords.js";
import type { Price } from "./market.js";
export type PylonError = {
    kind: "no-negation-boundary";
} | {
    kind: "malformed-srn";
    at: number;
} | {
    kind: "no-capacity";
    price: Price;
} | {
    kind: "boundary-rejects";
    frame: Coord;
} | {
    kind: "stalled";
    ticks: number;
    priceRising: boolean;
};
//# sourceMappingURL=errors.d.ts.map