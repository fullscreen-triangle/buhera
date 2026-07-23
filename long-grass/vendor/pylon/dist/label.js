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
/** FNV-1a 64-bit over a UTF-8 string, returned as hex. Stable, non-cryptographic. */
export function digestOf(data) {
    // 64-bit FNV-1a using BigInt for stability across platforms.
    let hash = 0xcbf29ce484222325n;
    const prime = 0x100000001b3n;
    const mask = 0xffffffffffffffffn;
    for (let i = 0; i < data.length; i++) {
        hash ^= BigInt(data.charCodeAt(i) & 0xff);
        hash = (hash * prime) & mask;
    }
    return { hex: hash.toString(16).padStart(16, "0") };
}
/**
 * A trajectory address — a path in the node's committed-unit tree. Advancing it
 * once per committed unit yields a strictly growing, replay-resistant address.
 */
export class TrajectoryAddress {
    branching;
    path = [];
    /** Monotone count of advances (== committed units addressed). */
    count = 0;
    constructor(branching = 3) {
        this.branching = branching;
    }
    static root(branching = 3) {
        return new TrajectoryAddress(branching);
    }
    /** Advance by one digit (0..branching-1). Monotone; never rewinds. */
    advance(digit) {
        this.path.push(((digit % this.branching) + this.branching) % this.branching);
        this.count++;
    }
    key() {
        return this.path.length === 0 ? "root" : this.path.join(".");
    }
}
//# sourceMappingURL=label.js.map