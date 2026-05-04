# Buhera — Long-Term Rust Implementation Plan

This document is the canonical roadmap for implementing the Buhera framework in Rust, anchored in the two formal papers in `long-grass/docs/` and the integration contract in `long-grass/integration.md`. The Python suites under `driven/src/utl/` and `driven/src/coe/` are **validation oracles**, not the implementation; they remain as a regression cross-check against the Rust kernel.

The plan is phased. Each phase ships a self-contained deliverable, ties to specific theorems from the papers, and has explicit acceptance criteria.

---

## 0. Position in the repository

### 0.1 What stays where

| Path | Role | Status under this plan |
|---|---|---|
| `mechanistic-synthesis/implementation/` | Purpose Rust workspace (vaHera AST, traits, Executor, CLI) | **Canonical** — all kernel work happens here as a new `purpose-kernel` crate. Stable interfaces in §2.1 of integration.md are frozen. |
| `long-grass/docs/os-throughput-law/` | UTL paper + 6 panels + 15 validation results | Reference. Each subsystem in this plan cites a specific UTL theorem. |
| `long-grass/docs/computational-operations-equivalence/` | COE paper + 6 panels + 15 validation results | Reference. Each subsystem cites a specific COE theorem. |
| `driven/src/utl/`, `driven/src/coe/` | Python validation oracles, 15 + 15 = 30/30 PASS | **Regression spec.** A Rust port (`buhera-conformance` crate, Phase 8) replays these experiments against the kernel. |
| `long-grass/src/` (Next.js) | Webtool / paper preview | Unchanged. Not in scope. |
| `src/`, top-level `Cargo.toml` (Buhera VPOS legacy) | Pre-formal speculation: consciousness substrate, gas oscillation, virtual foundry | **Open question — see §0.2.** |

### 0.2 What to do with the legacy top-level workspace

The top-level crate (`Cargo.toml` declaring `[package] name = "buhera"` with consciousness/foundry/gas-oscillation features) predates the formal grounding established by the UTL and COE papers. It contains modules whose claims are not derivable from the three axioms of either paper.

Three options, in order of recommendation:

1. **Move to `legacy/buhera-vpos/`** as a frozen archive. The module names remain searchable; nothing is lost; the top-level workspace becomes free for the new kernel.
2. **Leave it in place** but stop adding to it. New work goes only in `mechanistic-synthesis/implementation/`.
3. **Audit and reconcile.** For each legacy module, identify whether any of its content can be re-derived from UTL/COE axioms; salvage that, drop the rest.

This decision is yours; the rest of the plan assumes option 1 or 2 (i.e. nothing breaking the legacy code, but no new investment in it).

---

## 1. Theorem → Subsystem mapping

The integration contract identifies five Buhera subsystems. The two papers identify the theorems each must satisfy. The mapping is the load-bearing rationale for why these five subsystems and not others:

| Subsystem | Paper § | Theorem / experiment | What the subsystem must do |
|---|---|---|---|
| **PVE** Proof Validation Engine | COE Thm 5 (Three-Route Equivalence), V6 | $Q_\mathrm{I} = Q_\mathrm{II} = Q_\mathrm{III}$ for every operation | Reject any fragment whose three-route weights diverge; in release mode, reject only on the fast route. |
| **CMM** Categorical Memory Manager | UTL Thm 5 (Cache Extinction), V5 | $\tau$ structurally collapses to 0 when a decision class is constant-time | Hash-by-arguments cache; record per-class extinction events; expose the extinction rate as a UTL observable. |
| **PSS** Penultimate State Scheduler | UTL Thm 4 (Critical Slowing), V9–V10 | $\tau_\mathrm{relax} \sim |R - R_b|^{-1}$ near regime boundaries $\{0.3, 0.5, 0.8, 0.95\}$ | Estimate live $R$ from class-assignment series; order pending ops by distance to the closest regime boundary. |
| **DIC** Demon I/O Controller | UTL §6 (Coupling estimation), V14 | Empirical $g^{(ij)}$ recovers injected coupling at correlation $\geq 0.95$ | Surgical retrieval — fetch only bits whose mutual information with the query exceeds a coupling-derived threshold. |
| **TEM** Triple Equivalence Monitor | COE Thm 7 (MTIC), Thm 8 (Sliding-Endpoint), V7, V8, V9, V11 | $t = Q/f$, $M = Q$, monotone $M$, reproducibility ⇔ irreversibility | Independent sampler running at a slower cadence than dispatch; checks all four equivalences hold; alarms on divergence. |

Any subsystem we add later must cite a theorem in this table or in a successor paper. Subsystems without a theorem are not part of the kernel.

---

## 2. Stability contract for new code

The frozen interfaces in §2.1 of `integration.md` (`VaHera`, `Value`, `Type`, `Operation`, `Domain`, `Resolver`, `Provider`, `OperationRegistry::register`) are not modified by any phase of this plan. The kernel wraps the executor; it does not rewrite its inputs.

New traits introduced by this plan get their own stability contract:

| Symbol | Crate | Contract |
|---|---|---|
| `Subsystem` | `purpose-kernel` | One method `name()`, one async `tick()`. New methods may be added with a default impl. |
| `Cmm`, `Pss`, `Dic`, `Pve`, `Tem` | `purpose-kernel` | Each is a trait with a single, named role. Concrete implementations live in submodules; the trait surface is permanent once Phase 2 ships. |
| `Observable<T>` | `purpose-kernel` | Read-only handle for TEM and out-of-band consumers to subscribe to kernel-level invariants without coupling to internals. |

Versioning follows the workspace's `0.x.y` semver: each new public trait or struct method bumps the minor version; renames or removals are major.

---

## 3. Phase plan

Each phase has: deliverable crate(s) and feature(s); theorem coverage; acceptance criteria (what makes the phase done); rough size; depends-on.

### Phase 1 — Skeleton kernel

**Deliverable.** New crate `purpose-kernel` in `mechanistic-synthesis/implementation/crates/purpose-kernel/`. Defines `BuheraKernel`, the five `Subsystem` traits, and a `dispatch()` method that wraps `purpose_operations::Executor::execute()` with five no-op subsystems. CLI gains a `--kernel` flag (per integration.md §6.3, Stage 2).

**Theorem coverage.** None yet. This phase only proves the wiring.

**Acceptance.**
- All `purpose-domains-protein` operations dispatch through the kernel and produce identical results to direct `Executor` use.
- The CLI accepts `--kernel` and shows kernel telemetry (currently empty).
- `cargo test -p purpose-kernel` passes; no regression in existing crates.

**Size.** ~600 LOC across `lib.rs`, five trait files, integration test.

**Depends on.** Nothing — Purpose workspace already builds.

### Phase 2 — PVE first (Three-Route Equivalence)

**Deliverable.** `purpose-kernel::pve` module. Implements the three-route weight estimator (Residue, Confinement, Negation Fixed Point — exactly mirroring the Python `validate_03/04/05_route_*.py`), runs them in parallel on each fragment in debug builds, and rejects fragments whose three weights disagree.

In release builds, only the fast route (Residue) runs; PVE becomes a no-op gate that simply forwards the fragment.

**Theorem coverage.** COE Thm 5 (Three-Route Equivalence). COE V3, V4, V5, V6 enforced at dispatch.

**Acceptance.**
- A debug-build kernel rejects a hand-crafted fragment whose three routes diverge.
- A release-build kernel admits all three-route-consistent fragments at no measurable per-dispatch cost (< 1 µs overhead).
- A new conformance test mirrors Python `coe/validate_06_three_route_equivalence.py` and PASSes 50/50.

**Size.** ~1200 LOC.

**Depends on.** Phase 1.

### Phase 3 — CMM (Cache Extinction)

**Deliverable.** `purpose-kernel::cmm` module. Argument-hash cache keyed on `(op_name, canonicalised(args))`. Per-class hit-rate counters. A `CmmObservable` exposing extinction events as the structural collapse $\tau^{(ij)} \to 0$ described in UTL Thm 5.

The CMM does not store domain content directly; it stores `Value` results returned by providers. Cache-eviction policy is LRU with a configurable bound.

**Theorem coverage.** UTL Thm 5 (Cache Extinction). UTL V5 (extinction rate) becomes a live kernel metric.

**Acceptance.**
- Repeated identical calls hit the cache; metrics report nonzero hit rate.
- The structural drop in measured $\tau$ matches the predicted extinction relation to within $\pm 5\%$ on a synthetic workload.
- A conformance test mirrors Python `utl/validate_05_cache_extinction.py` and PASSes.

**Size.** ~900 LOC plus benchmarks.

**Depends on.** Phase 1.

### Phase 4 — PSS (Critical-Slowing-Aware Scheduling)

**Deliverable.** `purpose-kernel::pss` module. Maintains a sliding-window estimator of the phase coherence $R$ from the recent class-assignment series (mirrors Python `compute_R` in `utl/validate_06_phase_coherence.py`). When the kernel has multiple pending ops, PSS orders them by predicted distance to the nearest regime boundary $R_b \in \{0.3, 0.5, 0.8, 0.95\}$, dispatching the most slack-bearing one first.

PSS is opt-in: the kernel accepts a scheduling policy and falls back to FIFO if PSS is disabled.

**Theorem coverage.** UTL Thm 4 (Critical Slowing) + Algorithm `Load`. UTL V6, V7, V9, V10.

**Acceptance.**
- The estimator recovers $R$ from a synthetic class series with the same fidelity as the Python reference (correlation $\geq 0.99$).
- On a workload near a regime boundary, PSS reduces measured tail latency relative to FIFO by a measurable fraction (target: 20% on the conformance benchmark).
- A conformance test mirrors `utl/validate_10_load_indicator.py`; the alarm triggers correctly on $\geq 95\%$ of induced boundary-crossings.

**Size.** ~1500 LOC. Includes a small online-statistics utility.

**Depends on.** Phase 1.

### Phase 5 — TEM (Conservation Monitor)

**Deliverable.** `purpose-kernel::tem` module. Independent tokio task running at a configurable cadence (default: every 100 ms). Subscribes to dispatch events via `Observable<DispatchEvent>` and checks four invariants:

1. $M$ is monotone (rejects any decrement; alarms if observed).
2. $t \cdot f - M$ stays within tolerance (Time-Count Identity, COE V1).
3. Replays of an op produce identical $Q$ and identical hash (Reproducibility, COE V8).
4. Externally-induced truncation of $M$ breaks reproducibility (Sliding-Endpoint, COE V9).

Alarms surface as structured log events. TEM does not stop the kernel; it observes.

**Theorem coverage.** COE Thm 1 (Time-Count Identity), Thm 7 (MTIC), Thm 8 (Sliding-Endpoint), monotone-log corollary.

**Acceptance.**
- Synthetic violation of each invariant triggers exactly one alarm.
- TEM consumes < 1% CPU on a $10^4$-ops/s workload.
- Conformance tests mirror `coe/validate_07/08/09/11_*.py` and PASS.

**Size.** ~1100 LOC.

**Depends on.** Phase 1, Phase 2 (so that $Q$ values are trustworthy).

### Phase 6 — DIC (Surgical Retrieval)

**Deliverable.** `purpose-kernel::dic` module and a new provider kind `DemonProvider`. Given a query $A_Q$ and a candidate source $D$, DIC returns the bits with the highest mutual information $I(D; A_Q)$ above a coupling-derived threshold $g^*$.

DIC is opt-in per-provider; classical providers stay subject to the existing `Provider::invoke` contract unchanged.

**Theorem coverage.** UTL §6 (Coupling estimation), V14.

**Acceptance.**
- DIC retrieves only the high-$g$ bits on a synthetic source where ground truth is known; recall on those bits $\geq 0.95$.
- Kernel-level dispatch through DIC has bounded byte-overhead relative to baseline retrieval.
- A conformance test mirrors `utl/validate_14_coupling_estimator.py` and PASSes.

**Size.** ~1300 LOC.

**Depends on.** Phase 1, Phase 4 (PSS feeds the live $R$/coupling estimate).

### Phase 7 — Federation (Multi-Kernel Composition)

**Deliverable.** New crate `purpose-federation`. Hosts $n$ `BuheraKernel` instances and a thin coordinator that:

- Routes incoming fragments to the kernel with the most slack (lowest $R$ near a boundary).
- Reports composite $TP^{-1}_\mathrm{fed}$ via the closed-form $1 - TP^{-1}_\mathrm{fed}/\Sigma = \prod_i (1 - TP^{-1}_i/\Sigma)$.
- Exposes saturation telemetry as a federation-wide observable.

Federation does **not** require Zangalewa; it is an in-process or sidecar Rust composition. Zangalewa remains a separate, optional, peer system per integration.md §9.

**Theorem coverage.** UTL Thm 6 (Federation), V11 + V12.

**Acceptance.**
- Composite $TP^{-1}_\mathrm{fed}$ matches the closed form to floating-point precision on a synthetic federation of 2–8 kernels.
- Diminishing-returns property holds across $n \in \{1, \ldots, 20\}$ on the federation benchmark.

**Size.** ~1800 LOC.

**Depends on.** Phases 2–5 (kernel must be production-quality before composing).

### Phase 8 — `buhera-conformance` (Rust port of the validation suite)

**Deliverable.** New crate `buhera-conformance`. Reimplements all 30 experiments (15 UTL + 15 COE) in Rust against the Phase 1–7 kernel. Persists JSON to `target/conformance/`. A single command `cargo run -p buhera-conformance` reproduces the entire suite.

The Python suite remains as a reference; the Rust suite is the runtime check. Discrepancies between the two are bugs.

**Theorem coverage.** All 30 experiments.

**Acceptance.**
- 30/30 PASS in Rust.
- Per-experiment numerical agreement with Python within $\pm 10^{-9}$ on identity claims, $\pm 5\%$ on Monte Carlo claims.
- Total run-time on commodity hardware $< 30\,\mathrm{s}$ (slower than Python is acceptable; faster is preferred).

**Size.** ~3000 LOC.

**Depends on.** Phases 1–7.

### Phase 9 — Cross-architecture conformance (V15)

**Deliverable.** Conformance run on $\geq 4$ host configurations (different CPU, OS, Rust toolchain). All 30 experiments must PASS on each, with architecture-specific reference frequency $f$ recorded but not affecting any identity claim (per UTL Thm 7 and COE Thm 9).

**Theorem coverage.** UTL V15 + COE V15.

**Acceptance.**
- 30/30 PASS on each architecture.
- Per-architecture $f$ values logged; no architecture-specific code path needed.

**Size.** CI configuration only.

**Depends on.** Phase 8.

### Phase 10 — Documentation and downstream contract

**Deliverable.** A `KERNEL.md` in `mechanistic-synthesis/implementation/crates/purpose-kernel/` that documents the kernel from the perspective of a domain author: how to depend on it, how to observe its events, how to compose. Plus a `BUHERA-OS.md` peer of `integration.md` that reframes the contract now that the kernel is real.

The `Resolver` and `Provider` traits remain unchanged through this phase; the kernel's surface is the only new public API.

**Acceptance.**
- A new domain crate (template) wired against the kernel compiles and PASSes its own conformance test on day one of writing it.
- The integration contract in `long-grass/integration.md` references this plan and the new docs without contradiction.

**Size.** Docs only.

**Depends on.** Phase 8.

---

## 4. Cross-cutting concerns

### 4.1 Determinism

Every kernel subsystem must be deterministic given a seed. PSS, CMM, DIC: all RNGs are seeded from a kernel-level config. TEM observations are pure functions of the dispatch trace. Cross-architecture invariance (Phase 9) depends on this.

### 4.2 Error model

Errors flow up unchanged. `Error::Type` from PVE, `Error::Provider` from DIC's mutual-information estimator, `Error::Compile` from a malformed fragment — all use the existing `purpose-core::Error` enum. No silent fallback; no error swallowing.

### 4.3 Observability

Every subsystem exposes structured events through `Observable<T>`. TEM is the canonical consumer; external tooling (e.g. a Prometheus exporter, a tracing layer) plugs in via the same handle. The kernel does not bake in a particular metrics framework.

### 4.4 No Zangalewa coupling

This plan does not require Zangalewa. If Zangalewa is present (integration.md §9), it composes through `Provider`s as documented; the kernel's behaviour is unchanged. If absent, the kernel works on its own.

### 4.5 Async runtime

`tokio` only, single workspace-pinned version. TEM and DIC are `tokio::spawn`ed tasks with named lifetimes.

---

## 5. Non-goals

The following are explicitly outside the scope of this plan:

- A kernel scheduler for OS-level threads or processes. The "kernel" here is the Buhera dispatch kernel of integration.md §6, not Linux's.
- Any presentation/UI work. The Interceptor (integration.md §7) is a separate future crate.
- Hardware substrate work (consciousness substrate, gas oscillation, etc.). The legacy `src/` modules stay where they are.
- A training loop. Belongs to `purpose-factory`, not this plan.
- Cross-process federation transport. Phase 7 federates in-process or via a CLI subprocess; cross-host networking belongs to a successor.

---

## 6. Risks and exit criteria

### 6.1 Risks

- **PSS overhead.** A naive online $R$ estimator could dominate dispatch cost. Mitigation: amortise estimation over a window, gate it behind a feature flag, benchmark in Phase 4 before locking the API.
- **PVE three-route divergence in production.** Some operations may be hard to express through all three routes equally cheaply. Mitigation: in release builds, only the residue route runs; debug builds catch the divergence at test time.
- **Conformance drift between Python and Rust.** Floating-point reductions order-of-summation matters. Mitigation: where the experiment is identity-claim, both implementations use Kahan summation; where it is Monte Carlo, both use the same RNG (PCG64) seeded identically.
- **Legacy `src/` rot.** If the legacy workspace is left in place, builds may break unrelated to the kernel work. Mitigation: pin its dependencies and exclude it from CI's "must-pass" set.

### 6.2 Exit criteria for the plan

The plan is complete when:

1. `purpose-kernel` and `buhera-conformance` exist as published crates in the workspace.
2. All 30 experiments PASS in Rust on $\geq 4$ host configurations.
3. The `KERNEL.md` and `BUHERA-OS.md` documents are written and cross-referenced from `integration.md`.
4. A new domain crate can be written from scratch against the kernel without modifying any frozen interface.
5. The two papers (UTL, COE) are referenced from the kernel's `lib.rs` doc as the formal grounding, and from the conformance crate's `lib.rs` doc as the experimental specification.

---

## 7. Indicative timeline

The order of phases matters; the absolute pace does not. As a rough sketch (assuming one engineer working part-time):

| Phase | Working days |
|---|---|
| 1 — skeleton | 2 |
| 2 — PVE | 4 |
| 3 — CMM | 3 |
| 4 — PSS | 5 |
| 5 — TEM | 4 |
| 6 — DIC | 5 |
| 7 — federation | 5 |
| 8 — Rust conformance suite | 7 |
| 9 — cross-arch CI | 1 |
| 10 — docs | 2 |
| **Total** | **~38 working days** |

Phases 2 and 3 can run in parallel after Phase 1 ships. Phases 5 and 6 can run in parallel after Phase 4 ships.

---

## 8. Decision points before starting

These are the open questions whose answers shape the early phases. None of them blocks Phase 1.

1. **Legacy `src/`.** Move to `legacy/`, leave in place, or audit-and-reconcile? (See §0.2.)
2. **PSS feature gating.** Default-on or default-off in Phase 1's release build? (Recommendation: default-off until Phase 4 ships its benchmarks.)
3. **DIC's mutual-information backend.** Pure Rust (statrs / ndarray) or borrow from a domain crate? (Recommendation: pure Rust to keep the kernel self-contained.)
4. **Conformance suite location.** A peer crate `buhera-conformance` (recommended) or a `tests/` subdirectory of `purpose-kernel`? Peer crate keeps the kernel small and lets the suite ship as its own deliverable.
5. **Federation transport.** In-process `Arc<BuheraKernel>` only, or also a CLI-subprocess transport for the V15 cross-architecture test?

Once these are settled, Phase 1 can start.
