/**
 * Partition coordinates (n, l, m, s) — node identity in the SRN address space.
 *
 * Derived from SO(3) representation theory (SRN paper §2.2):
 *   n >= 1        — partition depth (shell)
 *   0 <= l < n    — structural complexity (angular-momentum index)
 *   -l <= m <= l  — orientation index
 *   s in {-1, +1} — residue parity (spin)
 *
 * Shell capacity: C(n) = 2n^2 (each shell n holds exactly 2n^2 valid addresses).
 *
 * Ported from `crates/srn-node/src/coords.rs` (the reference implementation).
 */
/** A validated partition coordinate. Immutable. */
export interface PartitionCoords {
    /** Depth / shell index (>= 1). */
    readonly n: number;
    /** Structural complexity (0 <= l < n). */
    readonly l: number;
    /** Orientation index (-l <= m <= l). */
    readonly m: number;
    /** Residue parity (+1 or -1). */
    readonly s: 1 | -1;
}
/** The public SRN partition-coordinate tuple form (n, l, m, s), per the API contract. */
export type Coord = readonly [number, number, number, number];
/** Error explaining why a coordinate quadruple is invalid. */
export interface CoordError {
    readonly kind: "invalid-coord";
    readonly message: string;
}
/**
 * Construct a validated PartitionCoords, enforcing the angular-momentum
 * constraints. Returns a typed error rather than throwing (core discipline).
 */
export declare function makeCoords(n: number, l: number, m: number, s: number): PartitionCoords | CoordError;
/** Type guard: did makeCoords succeed? */
export declare function isCoordError(x: PartitionCoords | CoordError): x is CoordError;
/** Build coords from a Coord tuple, or return the typed error. */
export declare function coordsFromTuple(c: Coord): PartitionCoords | CoordError;
/** The tuple form of a coordinate (for the public API surface). */
export declare function coordTuple(c: PartitionCoords): Coord;
/** The minimal-depth reference node (1, 0, 0, +1) — the "chromebook reference". */
export declare function referenceCoords(): PartitionCoords;
/** Shell capacity at depth n: C(n) = 2n^2. */
export declare function shellCapacity(n: number): number;
/** Enumerate all valid coordinates at depth n (the full shell). */
export declare function shell(n: number): PartitionCoords[];
/** Compact content-addressing key, e.g. "(2,1,0,+)". */
export declare function coordKey(c: PartitionCoords): string;
/** Structural equality of two coordinates. */
export declare function coordsEqual(a: PartitionCoords, b: PartitionCoords): boolean;
//# sourceMappingURL=coords.d.ts.map