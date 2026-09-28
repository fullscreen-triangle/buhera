# 01 — The Buhera Federation: overview

**Status:** normative · **Revision:** 2026-09-29 · **Scope:** the DSL registry, the module registry, and the ten engines integrated in this revision

## 1. What the federation is

Buhera is a categorical operating system whose capabilities come from **engines** that live in other repositories: a systems-biology shader, a federated query interpreter, a timing language, a repository-character invariant, a Turbulance runtime, and more. The federation is the machinery that makes those engines usable as one system without absorbing them:

- **One contract.** Every engine is exposed as a *module*: instruction in, act, ActResult out (spec 02).
- **One funnel.** Every act goes through `Registry::dispatch`, which audits it and notifies observers (spec 03).
- **One language table.** Every language a module owns is registered with its real compiler, the module that runs it, and the pack that grounds its generation (spec 04).
- **One catalogue.** A machine-readable list of members, checked by tests in both host languages (spec 05).
- **One sourcing rule.** Engines are vendored byte-exact from a recorded upstream commit, and drift is checked mechanically (spec 06).
- **Four bindings.** A host reaches an engine natively, remotely, by bridge, or not at all, and says which (spec 07).

The load-bearing rule underneath all of it is **M1: wrap, never reimplement.** A number a user sees came from the engine the paper describes. Where an engine cannot run on a host, the binding says so; nothing is approximated.

## 2. The two libraries

| | Rust | TypeScript |
|---|---|---|
| Contract, registry, DSL registry, catalogue | `buhera-os/crates/buhera-registry` | `buhera-os/registry-ts` (`@buhera/registry`) |
| Module adapters | `buhera-os/crates/buhera-modules` (one cargo feature per module) | `@buhera/registry/modules` (engines injected by the host) |
| Rust modules for the TS host | — | `buhera-os/crates/buhera-wasm` → `buhera_modules.wasm` |
| Engine store | `buhera-os/vendor/` | `long-grass/vendor/` |
| Conformance | `cargo test -p buhera-modules --features full` | `npm test` in `registry-ts` |

Both libraries are engine-free at their core. The Rust crate depends on serde and thiserror only. The TS package imports no engine; the host injects them. So neither can drag an engine's dependency tree into a consumer that did not ask for it.

## 3. Architecture

```mermaid
flowchart TB
  subgraph Upstream["Upstream repositories (sources of truth)"]
    H[hegel<br/>sbs · hfq · sbs Rust]
    Z[zangalewa<br/>zangalewa-dsl · interceptor]
    P[pylon<br/>ts]
    S[stella-lorraine<br/>tempus web lib]
    W[wind-tunnel<br/>wt-dsl]
    B[bloodhound<br/>tracker χ]
    K[kwasa-kwasa<br/>ndombolo-core]
  end
  VS{{scripts/vendor-sync.mjs<br/>byte-exact @ recorded commit}}
  Upstream --> VS
  VS --> RV[(buhera-os/vendor)]
  VS --> TV[(long-grass/vendor)]
  subgraph Rust["Rust host (buhera-os)"]
    RR[buhera-registry] --- RM[buhera-modules]
    RM --> RV
    GW[buhera-gateway<br/>/api/dispatch] --> RM
    WA[buhera-wasm] --> RM
  end
  subgraph TS["TypeScript host (long-grass)"]
    TR[@buhera/registry] --- TM[modules]
    TM --> TV
    TM -->|in-process| WA
    TM -->|HTTPS| GW
    TM -->|broker| ZB[zangalewa connect<br/>local agent]
  end
  CAT[(catalogue.json)] -. conformance .-> RR
  CAT -. conformance .-> TR
```

## 4. The members

| id | layer | language | Rust | TS | engine |
|---|---|---|---|---|---|
| `vahera` | language | vaHera | native | — (host-local in long-grass) | buhera-os |
| `ndombolo` | language | Turbulance | native | native (wasm) | kwasa-kwasa |
| `sbs` | science | SBS | — | native | hegel (JS) |
| `sbs-core` | science | — | native | remote | hegel (Rust) |
| `tempus` | science | Tempus | — | native | stella-lorraine (web) |
| `hfq` | coordination | HFQ | — | native | hegel (JS) |
| `pylon` | coordination | SRN | — | native | pylon (TS) |
| `windtunnel` | observation | .wt | native | native (wasm) | wind-tunnel |
| `tracker` | observation | — | native | native (wasm) | bloodhound |
| `zangalewa-dsl` | generation | — | native (feature) | remote (broker) | zangalewa |

Each `—` is deliberate and justified in the module's specification. The main reasons:

- an upstream Rust crate that does not compile (tempus);
- a Rust counterpart that implements a different model (pylon);
- no implementation in that language at all (hfq);
- a Rust crate with no DSL, which is therefore a separate module (sbs vs sbs-core).

## 5. How to read the specifications

- `architecture/02`–`05`: the contract, the registry and catalogue semantics. Normative, with RFC 2119 keywords.
- `architecture/06`–`08`: how engines are sourced, bound and hosted.
- `architecture/09`: the conformance matrix, which says what proves each claim and how to run it.
- `architecture/10`: findings, meaning every defect the integration surfaced upstream or in Buhera, and what was done about each.
- `specs/<module>.md`: one per member, with the same sections throughout (purpose, upstream, language, instructions, deltas, residue, bindings, side effects, hazards, conformance, upstream notes).
- `registry/catalogue.json` and `registry/vendor.json`: the machine-readable truth the tests check.
- `site/`: a Vite site that renders all of the above, with diagrams generated from the catalogue.
