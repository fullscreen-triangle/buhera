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
import { coordKey } from "./coords.js";
/** The live cell registry Gamma. */
export class Registry {
    cells = new Map();
    /** Total committed record count — monotone, never decremented. */
    totalCount = 0;
    /** Append a record for the given coordinates. The only mutation (append-only). */
    append(coords, record) {
        const key = coordKey(coords);
        let entries = this.cells.get(key);
        if (entries === undefined) {
            entries = [];
            this.cells.set(key, entries);
        }
        entries.push({ record, seq: entries.length });
        this.totalCount++;
    }
    /** The most recent successful (value) evaluation for these coords, or undefined. */
    fetchLatest(coords) {
        const entries = this.cells.get(coordKey(coords));
        if (entries === undefined)
            return undefined;
        for (let i = entries.length - 1; i >= 0; i--) {
            const e = entries[i];
            if (e.record.result.kind === "value")
                return e;
        }
        return undefined;
    }
    /** Full history for these coordinates. */
    fetchAll(coords) {
        return this.cells.get(coordKey(coords)) ?? [];
    }
    /** True if any successful evaluation exists for these coordinates. */
    hasCell(coords) {
        return this.fetchLatest(coords) !== undefined;
    }
    /** Number of live cells (coords with at least one record). */
    cellCount() {
        return this.cells.size;
    }
}
//# sourceMappingURL=registry.js.map