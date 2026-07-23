/**
 * Live cell registry Gamma — content-addressed by partition coordinates.
 *
 * Gamma is the local, append-biased store of evaluated glyphs: coord key ->
 * append-only list of EvalRecord (the log never shrinks). "Live" means it
 * reflects this node's current evaluation state; it is NOT a global shared store
 * (SRN §6: every registry is local). Every node has its own Gamma.
 *
 * Ported from `crates/srn-node/src/registry.rs`.
 */
import { type PartitionCoords } from "./coords.js";
import type { EvalRecord } from "./label.js";
/** A single registry entry with its sequential index in the coord's history. */
export interface RegistryEntry {
    readonly record: EvalRecord;
    /** 0-based, monotone within this coord's history. */
    readonly seq: number;
}
/** The live cell registry Gamma. */
export declare class Registry {
    private readonly cells;
    /** Total committed record count — monotone, never decremented. */
    totalCount: number;
    /** Append a record for the given coordinates. The only mutation (append-only). */
    append(coords: PartitionCoords, record: EvalRecord): void;
    /** The most recent successful (value) evaluation for these coords, or undefined. */
    fetchLatest(coords: PartitionCoords): RegistryEntry | undefined;
    /** Full history for these coordinates. */
    fetchAll(coords: PartitionCoords): ReadonlyArray<RegistryEntry>;
    /** True if any successful evaluation exists for these coordinates. */
    hasCell(coords: PartitionCoords): boolean;
    /** Number of live cells (coords with at least one record). */
    cellCount(): number;
}
//# sourceMappingURL=registry.d.ts.map