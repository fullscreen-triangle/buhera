/* ============================================================================
 * @buhera/registry — the TypeScript registry library.
 *
 * Twin of the Rust crates buhera-registry (contract, registry, DSL registry,
 * catalogue) and buhera-modules (adapters). Normative text lives in
 * specifications/architecture/02..05 and specifications/specs/<module>.md.
 * Module adapters are exported separately from "@buhera/registry/modules" so
 * a host that only needs the contract never loads an engine.
 * ========================================================================== */

export * from "./contract.ts";
export * from "./registry.ts";
export * from "./dsl.ts";
export * from "./catalogue.ts";
export * from "./wasm.ts";
