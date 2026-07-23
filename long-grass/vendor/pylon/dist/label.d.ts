/**
 * Content-addressing labels and committed evaluation records.
 *
 * Every committed unit is annotated with a paired (digest, address) label:
 *   - digest: a content hash of the expression (content addressing);
 *   - address: the node's trajectory-address path at commit time.
 *
 * Ported from `crates/srn-node/src/label.rs`. Uses a small pure-JS FNV-1a hash
 * (no crypto dependency; content addressing here needs stability, not security —
 * SRN's structural incorruptibility comes from the absence of a parser, not from
 * a cryptographic primitive).
 */
import type { ReceiverFrame, EvalResult } from "./srn/expr.js";
/** A stable content digest of some bytes. */
export interface ContentDigest {
    readonly hex: string;
}
/** FNV-1a 64-bit over a UTF-8 string, returned as hex. Stable, non-cryptographic. */
export declare function digestOf(data: string): ContentDigest;
/**
 * A trajectory address — a path in the node's committed-unit tree. Advancing it
 * once per committed unit yields a strictly growing, replay-resistant address.
 */
export declare class TrajectoryAddress {
    private readonly branching;
    private readonly path;
    /** Monotone count of advances (== committed units addressed). */
    count: number;
    constructor(branching?: number);
    static root(branching?: number): TrajectoryAddress;
    /** Advance by one digit (0..branching-1). Monotone; never rewinds. */
    advance(digit: number): void;
    key(): string;
}
/** A paired process label for one committed unit. */
export interface ProcessLabel {
    readonly digest: ContentDigest;
    readonly address: string;
}
/** A committed evaluation record — one entry in the append-only eval log. */
export interface EvalRecord {
    readonly exprDigest: string;
    readonly address: string;
    readonly frame: ReceiverFrame;
    readonly result: EvalResult;
    readonly timestampNs: number;
}
//# sourceMappingURL=label.d.ts.map