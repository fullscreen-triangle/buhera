/**
 * SRN source parser.
 *
 * Parses the glyph grammar:
 *   |name : (n,l,m,s)| not { guard } do { body } to { target } [as { alias }]
 *
 * Enforces the Mandatory Negation Boundary (SRN Cor 2.1): a glyph with no `not`
 * clause is not individuated and is rejected. Returns a typed ParseError rather
 * than throwing (core discipline), so callers can surface
 * `no-negation-boundary` / `malformed-srn` cleanly.
 */
import { type Glyph } from "./expr.js";
/** Why a source string failed to parse. */
export type ParseError = {
    kind: "no-negation-boundary";
} | {
    kind: "malformed-srn";
    at: number;
    message: string;
};
/** True if the parse produced an error rather than a glyph. */
export declare function isParseError(x: Glyph | ParseError): x is ParseError;
/**
 * Parse a single SRN glyph expression from source text.
 *
 * A missing `not { ... }` clause yields { kind: "no-negation-boundary" }; any
 * other structural failure yields { kind: "malformed-srn", at, message }.
 */
export declare function parseSrn(source: string): Glyph | ParseError;
//# sourceMappingURL=parse.d.ts.map