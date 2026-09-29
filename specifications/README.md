# Buhera Federation — Specifications

The formal specification of how Buhera's modules and languages are integrated: the module contract, the module registry, the DSL registry, the catalogue, and one specification per member. It also covers the two libraries that implement them, the Rust `buhera-registry`/`buhera-modules` and the TypeScript `@buhera/registry`.

| | |
|---|---|
| `architecture/01-overview.md` | What the federation is; the two libraries; the member table |
| `architecture/02-module-contract.md` | Instruction, ActResult, Descriptor, Module; rules I, A, D, M |
| `architecture/03-registry.md` | Dispatch, audit log, hooks; rules R1–R6 |
| `architecture/04-dsl-registry.md` | Language → real validator, executing module, grounding pack; rules L1–L7 |
| `architecture/05-catalogue.md` | The normative member list; conformance rules C1–C5 |
| `architecture/06-sourcing.md` | Vendoring at a recorded commit; drift checks |
| `architecture/07-bindings.md` | native / remote / bridge / none; the gateway protocol; Rust in the browser via wasm |
| `architecture/08-hosts.md` | long-grass, buhera-gateway, the wasm module, the scheduler |
| `architecture/09-conformance.md` | How to run every check; results; claim → proof |
| `architecture/10-findings.md` | Defects found and fixed, and upstream change requests |
| `architecture/11-survey.md` | The second survey: every candidate examined across the repositories, its disposition, and what would unblock it |
| `specs/*.md` | vahera · ndombolo · sbs · sbs-core · tempus · hfq · pylon · windtunnel · tracker · zangalewa-dsl · smith · synopsis · cfc · sthurbert · honjo · heihachi · olduvai · levinthal · shapeshifter · ladder · spectral · mekaneck · scope |
| `registry/catalogue.json` | Machine-readable members; checked by both libraries' tests |
| `registry/vendor.json` | Every vendored engine copy and its upstream commit |
| `site/` | Vite site rendering all of this, with diagrams drawn from the catalogue |

## The site

```sh
cd specifications/site
npm install
npm run dev      # http://localhost:5173
npm run build    # static build in site/dist
```
