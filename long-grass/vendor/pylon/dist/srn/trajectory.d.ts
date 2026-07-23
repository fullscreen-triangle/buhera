/**
 * Timing-trajectory transmission codec (SRN paper §7).
 *
 * SRN's transmission unit is not bytes but a timing trajectory: a sequence of
 * arrival-timing deviations Delta P(k) across d channels. The partition
 * coordinates (n, l, m, s) and a body digest are encoded into that trajectory,
 * and decoded back at the receiver. The paper fixes the SHAPE (SRN §7); the
 * numeric scheme is an implementation choice (contract §13). We use a simple,
 * injective integer-lattice scheme:
 *
 *   channel 0 carries n and l (as two deviations),
 *   channel 1 carries m and the spin parity s,
 *   channel 2 carries the low bits of the body digest,
 *   (further channels carry more digest bits as needed).
 *
 * Injectivity (SRN Thm 7.x): distinct (coords, bodyDigest) -> distinct
 * trajectories, so decode(encode(x)) = x.
 */
import { type PartitionCoords } from "../coords.js";
import { type Glyph } from "./expr.js";
/** A timing trajectory: per-channel deviation sequences. */
export interface Trajectory {
    /** Flat list of timing deviations Delta P(k). */
    readonly deltas: number[];
    /** Number of channels d the deltas are laid out across. */
    readonly channels: number;
}
/** Encode partition coordinates + a body digest into a timing trajectory. */
export declare function encodeCoords(coords: PartitionCoords, bodyDigestHex: string): Trajectory;
/** Decode a timing trajectory back to coordinates + digest low-word. */
export declare function decodeCoords(traj: Trajectory): {
    coords: PartitionCoords;
    digestLow: number;
} | {
    error: string;
};
/**
 * Encode a full glyph as a timing trajectory: its target coordinates plus a
 * digest of its (notGuard, body, toTarget) so distinct glyphs at the same
 * address still map to distinct trajectories (injectivity).
 */
export declare function encodeTrajectory(g: Glyph): Trajectory;
/**
 * Decode a trajectory back to a glyph shell. The body text is not recoverable
 * from the digest (one-way), so the decoded glyph carries the digest as its body
 * marker; a receiver matches it against its registry by (coords, digest). This
 * mirrors SRN §7: the receiver recovers (n,l,m,s) exactly and looks up / matches
 * the expression, rather than reconstructing arbitrary source text.
 */
export declare function decodeTrajectory(traj: Trajectory): Glyph | {
    error: string;
};
//# sourceMappingURL=trajectory.d.ts.map