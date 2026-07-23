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
import { coordsFromTuple, isCoordError } from "../coords.js";
import { glyph } from "./expr.js";
import { digestOf } from "../label.js";
/** Quantum used to embed integers as timing deviations (arbitrary positive unit). */
const Q = 1e-6;
/** Encode partition coordinates + a body digest into a timing trajectory. */
export function encodeCoords(coords, bodyDigestHex) {
    const channels = 4;
    // spin parity mapped to {0,1}; low 32 bits of the digest split across two slots
    const digestLow = parseInt(bodyDigestHex.slice(-8), 16) >>> 0;
    const dHi = Math.floor(digestLow / 0x10000);
    const dLo = digestLow % 0x10000;
    const raw = [
        coords.n,
        coords.l,
        coords.m + 1024, // bias so negative m encodes as a positive deviation
        coords.s > 0 ? 1 : 0,
        dHi,
        dLo,
    ];
    return { deltas: raw.map((v) => v * Q), channels };
}
/** Decode a timing trajectory back to coordinates + digest low-word. */
export function decodeCoords(traj) {
    const r = traj.deltas.map((d) => Math.round(d / Q));
    if (r.length < 6)
        return { error: "trajectory too short" };
    const [n, l, mBias, sBit, dHi, dLo] = r;
    const tuple = [n, l, mBias - 1024, sBit === 1 ? 1 : -1];
    const coords = coordsFromTuple(tuple);
    if (isCoordError(coords))
        return { error: coords.message };
    return { coords, digestLow: dHi * 0x10000 + dLo };
}
/**
 * Encode a full glyph as a timing trajectory: its target coordinates plus a
 * digest of its (notGuard, body, toTarget) so distinct glyphs at the same
 * address still map to distinct trajectories (injectivity).
 */
export function encodeTrajectory(g) {
    const bodyDigest = digestOf(`${g.name}|${g.notGuard}|${g.body}|${g.toTarget}`).hex;
    return encodeCoords(g.target, bodyDigest);
}
/**
 * Decode a trajectory back to a glyph shell. The body text is not recoverable
 * from the digest (one-way), so the decoded glyph carries the digest as its body
 * marker; a receiver matches it against its registry by (coords, digest). This
 * mirrors SRN §7: the receiver recovers (n,l,m,s) exactly and looks up / matches
 * the expression, rather than reconstructing arbitrary source text.
 */
export function decodeTrajectory(traj) {
    const d = decodeCoords(traj);
    if ("error" in d)
        return d;
    return glyph({
        name: "decoded",
        target: d.coords,
        notGuard: "decoded", // a decoded glyph is individuated by its recovered address
        body: `digest:${d.digestLow.toString(16)}`,
        toTarget: "*",
    });
}
//# sourceMappingURL=trajectory.js.map