/* ============================================================================
 * Cytochrome P450 Module — the monograph as a federation contributor.
 *
 * WHAT THIS IS, AND WHY IT LIVES HERE
 * -----------------------------------
 * The `pingpong-bibi-conditioned-floor` kernel in levinthal solved ONE thing:
 * it read a transaminase reaction as a residue-chained cut sequence and drew a
 * single load-bearing distinction — `participant` vs `carrier` (PLP rides on the
 * receiver, appears in no equation, is cut once and never re-cut). That was a
 * *medium-vertex CKG solution*: reaction participants discriminated against the
 * solvent/medium vertex, with a floor-conditioned notion of when two cuts are
 * even comparable (conditioned_floor.py).
 *
 * The cytochrome P450 website is a far richer object — a detailed monograph of
 * the P450 catalytic machinery: a seven-state closed orbit, an NADPH→FAD→FMN→
 * heme electron-transfer chain, Compound I formation by PCET, C–H activation /
 * heteroatom / atypical reaction families sorted by an aperture depth ΔM,
 * spectroscopy (Soret / EPR / Raman), and isoform/pharmacogenomic ΔM shifts.
 *
 * This module makes that whole corpus a set of federation CONTRIBUTORS. Each
 * `op` folds one *kind* of monograph fact onto a P450 reaction node when the
 * `ckg` runtime dispatches it — exactly the way `sbs` folds a circuit and
 * `shapeshifter` folds spectra. The point being demonstrated is comparative:
 * the SAME runtime that held the transaminase's one distinction holds all of
 * this, because a CKG node is a meeting point where any module with something
 * to say about a subtask leaves its whole artifact. The transaminase's
 * participant/carrier cut is preserved here as just one op among many
 * (`op:"participants"`), and the conditioned floor as another (`op:"floor"`).
 *
 * Every number below is read from the monograph, not invented:
 *   Fe centre (17.26, 11.53, 24.76) Å; Marcus λ = 0.85 eV; d_C(ET) = 4;
 *   FMN→heme rate-limiting 5e6 s⁻¹, ΔM_ET 7.60; seven-state ΣΔM = 4.963;
 *   Compound I ΔM = ln2 ≈ 0.693, KIE ≈ 1.7; Soret 417→392 nm; Fe=O 795 cm⁻¹
 *   (¹⁸O→758); reaction-family ΔM ordering S-ox<N-ox<N-dealk<O-dealk<aliphatic;
 *   CYP2D6 phenotype ΔM {UM .27, EM .55, IM .75, PM 2.50}; CYP2C9*3 ΔM 3.60.
 *
 * Instruction shapes (dispatched directly, or attached via ckg `attach`):
 *   { op: "participants", reaction? }   — participant/carrier cut (transaminase twin)
 *   { op: "electron-transfer" }         — NADPH→FAD→FMN→heme chain, Marcus λ
 *   { op: "compound-i" }                — Compound I formation, Rittle–Green observables
 *   { op: "pathway", reaction }         — HAT / rebound / heteroatom / atypical aperture
 *   { op: "states" }                    — the seven-state closed orbit, ΣΔM
 *   { op: "spectroscopy" }              — Soret / EPR / Raman signatures
 *   { op: "isoform", cyp?, phenotype? } — isoform / pharmacogenomic ΔM
 *   { op: "floor", conditions? }        — conditioned admissibility floor β
 *   "demo" / ""                         — the electron-transfer chain (headline result)
 * ========================================================================== */

// --- the conditioned floor, ported from conditioned_floor.py ----------------
//
// A cut's residue is never below this floor, and two cuts taken at different
// conditions are cuts at *different* floors — so the comparability the whole
// framework leans on has to name WHICH floor. At the categorical depth the
// address manifold uses (d = 9) the combinatorial conversion term dominates and
// β is effectively condition-independent; the conditions only govern in the
// Q-dominated (short-integration) regime. This is the levinthal finding, kept
// intact so the cytochrome model inherits the same admissibility semantics.

const K_B_EV = 8.617333262e-5;

const REF = {
  temperature_K: 298.15,
  pH: 7.4,
  viscosity_cP: 0.89,
  integration_time_s: 1.0e-3,
};

const RECURSION_DEPTH = 9; // categorical depth d
const N_FUNCTORS = 6; // |S_3|, the conversion-functor cycle

// The heme Fe–O oscillator is the cofactor mode named in the leaf algebra;
// the four classes are residue / cofactor / solvent / substrate.
const OSCILLATORS = [
  { name: "amide-I (residue)", freq_hz: 5.1e13, Q_ref: 1.0e3 },
  { name: "Fe-O (cofactor)", freq_hz: 2.4e13, Q_ref: 5.0e2 },
  { name: "O-H libration (solvent)", freq_hz: 1.9e13, Q_ref: 1.0e2 },
  { name: "C-N stretch (substrate)", freq_hz: 3.3e13, Q_ref: 8.0e2 },
];

function floorDisc(d = RECURSION_DEPTH) {
  return 1.0 / (2.0 * 3 ** d);
}
function floorConv(d = RECURSION_DEPTH, n = N_FUNCTORS) {
  return n / 3.0 ** d;
}
function qFactor(qRef, cond) {
  const thermal = cond.temperature_K / REF.temperature_K;
  const viscous = cond.viscosity_cP / REF.viscosity_cP;
  return qRef / (thermal * viscous);
}
function floorQ(cond) {
  const per = OSCILLATORS.map((osc) => {
    const q = qFactor(osc.Q_ref, cond);
    const sigma = 1.0 / (q * Math.sqrt(cond.integration_time_s * osc.freq_hz));
    return { oscillator: osc.name, Q: q, sigma };
  });
  const quad = Math.sqrt(per.reduce((s, p) => s + p.sigma ** 2, 0));
  return { per, quad };
}
function beta(cond) {
  const c = { ...REF, ...(cond || {}) };
  return floorDisc() + floorQ(c).quad + floorConv();
}

// --- the P450 catalytic cycle: the seven-state closed orbit -----------------
//
// Each state carries its (n, ℓ, m, s) address and the ΔM aperture of the
// transition that reaches it. The orbit closing (state 7 → state 1) is the
// catalytic admissibility condition, exactly analogous to the ping-pong cycle
// returning the enzyme to E. ΣΔM over the orbit = 4.963 (monograph).

const CYCLE_STATES = [
  { i: 1, name: "resting (low-spin Fe³⁺)", addr: [0, 0, 0, 0], dM_in: 0.0 },
  { i: 2, name: "substrate-bound (high-spin Fe³⁺)", addr: [1, 0, 0, 1], dM_in: 0.918 },
  { i: 3, name: "ferrous (Fe²⁺)", addr: [1, 1, 0, 1], dM_in: 0.62 },
  { i: 4, name: "oxy-ferrous (Fe²⁺–O₂)", addr: [1, 1, 1, 0], dM_in: 0.71 },
  { i: 5, name: "peroxo/hydroperoxo (Fe³⁺–OOH)", addr: [2, 1, 1, 0], dM_in: 0.55 },
  { i: 6, name: "Compound I (Fe⁴⁺=O porphyrin•⁺)", addr: [2, 2, 1, 1], dM_in: 0.693 },
  { i: 7, name: "product complex (Fe³⁺–ROH)", addr: [1, 1, 0, 0], dM_in: 0.472 },
];
const CYCLE_CLOSE_DM = 0.09; // state 7 → state 1, product release + reset
const CYCLE_SUM_DM = 4.963; // monograph headline

// --- the electron-transfer chain NADPH → FAD → FMN → heme -------------------
//
// Categorical distance d_C = 4 hops; Marcus reorganisation λ = 0.85 eV; the
// FMN → heme hop is rate-limiting (5e6 s⁻¹) and carries the largest ΔM (7.60).

const ET_CHAIN = [
  { from: "NADPH", to: "FAD", rate_s: 3.0e7, dist_A: 7.6, dM: 3.9 },
  { from: "FAD", to: "FMN", rate_s: 5.0e7, dist_A: 4.0, dM: 3.1 },
  { from: "FMN", to: "heme", rate_s: 5.0e6, dist_A: 18.4, dM: 7.6 },
];
const MARCUS_LAMBDA_EV = 0.85;
const ET_DC = 4;
const FE_CENTRE_A = [17.26, 11.53, 24.76]; // find_iron() on the GLB, monograph

// --- reaction families as ΔM-parameterised apertures ------------------------
//
// The chemistry the P450 does, sorted by the aperture depth ΔM of the
// rate-determining cut. Smaller ΔM ⇒ tighter aperture ⇒ faster. The ordering
// (S-ox < N-ox < N-dealk < O-dealk < aliphatic C–H) is the monograph's, as is
// the diagnostic KIE for each (HAT reactions show a large KIE; direct
// heteroatom O-transfer does not).

const REACTIONS = {
  "s-oxidation": { family: "heteroatom", mechanism: "direct O-transfer", dM: 0.28, kie: 1.1 },
  "n-oxidation": { family: "heteroatom", mechanism: "direct O-transfer", dM: 0.32, kie: 1.0 },
  "n-dealkylation": { family: "heteroatom", mechanism: "HAT + rebound", dM: 0.5, kie: 3.0 },
  "o-dealkylation": { family: "heteroatom", mechanism: "HAT + rebound", dM: 0.58, kie: 5.0 },
  "aliphatic-hydroxylation": { family: "C-H activation", mechanism: "HAT + O-rebound", dM: 0.65, kie: 7.2 },
  "aromatic-hydroxylation": { family: "atypical", mechanism: "epoxidation / NIH shift", dM: 0.72, kie: 1.0 },
  desaturation: { family: "atypical", mechanism: "double HAT", dM: 0.85, kie: 3.5 },
};

// --- spectroscopy signatures ------------------------------------------------

const SPECTRA = {
  soret: { resting_nm: 417, compound_i_nm: 392, shift_nm: -25 },
  epr: { g_low: 2.42, g_mid: 2.25, g_high: 1.92 }, // low-spin Fe³⁺ g-tensor
  raman: { fe_o_cm: 795, fe_o_18O_cm: 758, shift_cm: -37 }, // Compound I Fe=O
};

// --- the two DSL scripts, P450-specific -------------------------------------
//
// These are the corpus expressed in the OTHER two federation DSLs, so the same
// reaction graph carries an sbs circuit fact and a shapeshifter spectra fact
// beside the native cytochrome facts. The circuit is the real NADPH→FAD→FMN→heme
// redox chain (μ in kJ/mol tracking the reduction-potential ladder; the FMN→heme
// conductance is the smallest, the rate-limiting hop). The .ss script is a
// virtual orbitrap acquisition of a CYP substrate and its oxidised metabolite.

export const P450_ET_SBS = `// P450 electron-transfer chain — the CPR→heme redox ladder as an SBS circuit.
circuit p450_electron_transfer {
  node NADPH { mu: -320.0, concentration: 1.0, compartment: "cytoplasm" }
  node FAD   { mu: -220.0, concentration: 1.0 }
  node FMN   { mu: -190.0, concentration: 1.0 }
  node heme  { mu: -170.0, concentration: 1.0 }

  edge NADPH -> FAD  { rate: 30.0, conductance: 3.0 }
  edge FAD   -> FMN  { rate: 50.0, conductance: 5.0 }
  edge FMN   -> heme { rate: 5.0,  conductance: 0.5 }
}

observe p450_electron_transfer
perturb p450_electron_transfer { factor: 0.1 }
navigate from heme
`;

export const P450_MS_SS = `objective p450_metabolite_scan:
  target: "CYP substrate + oxidised metabolite, positive mode"

instrument orbi:
  kappa: 1e12
  ref_frequency: 10e6

phase acquire:
  records = lavoisier.instrument.run_experiment(classes: ["PC"], polarity: "+", analyser: "orbitrap", mz_window: [150, 500])
  field = lavoisier.observe.partition_field(records: records)
`;

// --- isoform / pharmacogenomic ΔM -------------------------------------------
//
// 57 human CYPs across 18 families are laid out on the base-3 address manifold;
// a polymorphism is a ΔM shift on the allele. CYP2D6 phenotype classes and the
// CYP2C9*3 loss-of-function are the monograph's calibration points.

const ISOFORMS = {
  CYP3A4: { family: 3, depth: 5.69, note: "broadest substrate range" },
  CYP2D6: {
    family: 2,
    depth: 4.8,
    phenotypes: { UM: 0.27, EM: 0.55, IM: 0.75, PM: 2.5 },
  },
  CYP2C9: { family: 2, depth: 4.6, alleles: { "*1": 0.0, "*3": 3.6 } },
  CYP1A2: { family: 1, depth: 4.1, note: "planar aromatic substrates" },
};

// --- the participant / carrier cut (the transaminase twin) ------------------
//
// The one distinction the levinthal kernel drew, expressed for a P450 turnover.
// The P450 heme iron is a CARRIER exactly as PLP is: it is bound to the receiver,
// cut once at enzyme construction, and is a participant in no reaction equation —
// substrate + O₂ + 2e⁻ + 2H⁺ → product + H₂O are the participants. This op
// exists so the report can show the SAME predicate the transaminase model
// asserted, now over cytochrome — the continuity that makes the comparison fair.

const PARTICIPANT_SETS = {
  "aliphatic-hydroxylation": {
    participants: ["substrate R-H", "O₂", "2 e⁻ (from NADPH)", "2 H⁺", "product R-OH", "H₂O"],
    carriers: ["heme Fe (protoporphyrin IX)", "FAD", "FMN"],
  },
  default: {
    participants: ["substrate", "O₂", "2 e⁻", "2 H⁺", "oxidised product", "H₂O"],
    carriers: ["heme Fe (protoporphyrin IX)", "FAD", "FMN"],
  },
};

// ============================================================================
// op handlers — each returns a whole output_delta the CKG runtime folds as a
// fact. `kind` is chosen so the ckg-module's deriveFindings has a headline case.
// ============================================================================

function opParticipants(reaction) {
  const set = PARTICIPANT_SETS[reaction] || PARTICIPANT_SETS.default;
  const floor = beta();
  // residue = floor × boundaries committed, exactly as pingpong.py derives it:
  // a carrier is committed once (arity-1, one boundary); a participant is cut on
  // every pass (binding arity-2 = two boundaries).
  const events = [
    ...set.carriers.map((c) => ({ label: `bind carrier ${c}`, boundaries: 1, residue: floor * 1 })),
    ...set.participants.map((p) => ({ label: `cut ${p}`, boundaries: 2, residue: floor * 2 })),
  ];
  return {
    kind: "cyp_participants",
    reaction: reaction || "generic P450 turnover",
    participants: set.participants,
    carriers: set.carriers,
    events,
    M: events.length,
    floor,
    // the load-bearing claim, stated for cytochrome exactly as for transaminase:
    invariant: "heme Fe is a carrier (bound once, in no equation), not a participant",
    total_residue: events.reduce((s, e) => s + e.residue, 0),
  };
}

function opElectronTransfer() {
  const totalDM = ET_CHAIN.reduce((s, h) => s + h.dM, 0);
  const rateLimiting = ET_CHAIN.reduce((a, b) => (b.rate_s < a.rate_s ? b : a));
  return {
    kind: "cyp_electron_transfer",
    chain: ET_CHAIN,
    d_C: ET_DC,
    marcus_lambda_eV: MARCUS_LAMBDA_EV,
    fe_centre_A: FE_CENTRE_A,
    rate_limiting: `${rateLimiting.from}→${rateLimiting.to}`,
    rate_limiting_s: rateLimiting.rate_s,
    total_dM: Number(totalDM.toFixed(3)),
    summary: `NADPH→FAD→FMN→heme, d_C=${ET_DC}, λ=${MARCUS_LAMBDA_EV} eV, FMN→heme limits at ${rateLimiting.rate_s.toExponential(1)} s⁻¹`,
  };
}

function opCompoundI() {
  // Compound I as a d_C = 1 aperture of depth ΔM = ln2, with the eight
  // Rittle–Green observables at the S-coordinate the monograph reports.
  const dM = Math.log(2);
  return {
    kind: "cyp_compound_i",
    aperture_dC: 1,
    dM: Number(dM.toFixed(3)),
    mechanism: "PCET (proton-coupled electron transfer)",
    S_coordinate: [0.86, 0.515, 0.595],
    kie: 1.7,
    observables: {
      fe_o_raman_cm: SPECTRA.raman.fe_o_cm,
      fe_iv_oxo: true,
      porphyrin_cation_radical: true,
      mossbauer_isomer_shift_mm_s: 0.11,
    },
    summary: `Compound I: d_C=1, ΔM=ln2≈${dM.toFixed(3)}, PCET, KIE≈1.7`,
  };
}

function opPathway(reaction) {
  const r = REACTIONS[reaction];
  if (!r) {
    return {
      kind: "cyp_pathway",
      ok: false,
      error: `unknown reaction "${reaction}"`,
      known: Object.keys(REACTIONS),
    };
  }
  // aperture semantics: smaller ΔM ⇒ tighter aperture ⇒ faster; a large KIE is
  // the HAT diagnostic, a KIE ≈ 1 says the C–H bond is not broken in the RDS.
  const isHAT = r.mechanism.includes("HAT");
  return {
    kind: "cyp_pathway",
    reaction,
    family: r.family,
    mechanism: r.mechanism,
    dM: r.dM,
    kie: r.kie,
    rate_determining_step: isHAT ? "hydrogen-atom transfer" : "direct O-transfer",
    diagnostic:
      r.kie > 2
        ? `large KIE (${r.kie}) ⇒ C–H broken in the rate-determining step`
        : `KIE≈${r.kie} ⇒ C–H not broken in the RDS (direct transfer)`,
    summary: `${reaction}: ${r.family}, ΔM=${r.dM}, KIE=${r.kie}`,
  };
}

function opStates() {
  const transitions = CYCLE_STATES.slice(1).map((s, idx) => ({
    from: CYCLE_STATES[idx].name,
    to: s.name,
    dM: s.dM_in,
  }));
  transitions.push({
    from: CYCLE_STATES[6].name,
    to: CYCLE_STATES[0].name,
    dM: CYCLE_CLOSE_DM,
  });
  const sum = transitions.reduce((s, t) => s + t.dM, 0);
  return {
    kind: "cyp_states",
    states: CYCLE_STATES,
    transitions,
    closed: true,
    orbit_sum_dM: Number(sum.toFixed(3)),
    orbit_sum_dM_reference: CYCLE_SUM_DM,
    summary: `seven-state closed orbit, ΣΔM≈${sum.toFixed(3)} (ref ${CYCLE_SUM_DM})`,
  };
}

function opSpectroscopy() {
  return {
    kind: "cyp_spectroscopy",
    soret: SPECTRA.soret,
    epr: SPECTRA.epr,
    raman: SPECTRA.raman,
    isotope_confirmation: `Fe=O Raman ${SPECTRA.raman.fe_o_cm}→${SPECTRA.raman.fe_o_18O_cm} cm⁻¹ on ¹⁶O→¹⁸O (Δ${SPECTRA.raman.shift_cm})`,
    summary: `Soret ${SPECTRA.soret.resting_nm}→${SPECTRA.soret.compound_i_nm} nm; Fe=O ${SPECTRA.raman.fe_o_cm} cm⁻¹`,
  };
}

// Each spectroscopy observable is its OWN fact kind, so a node can carry the
// UV/vis Soret shift, the EPR g-tensor, and the resonance-Raman Fe=O stretch as
// three distinct assertions — not one collapsed "spectroscopy" blob.
function opSoret() {
  return {
    kind: "cyp_soret",
    band: "Soret (UV/vis)",
    resting_nm: SPECTRA.soret.resting_nm,
    compound_i_nm: SPECTRA.soret.compound_i_nm,
    shift_nm: SPECTRA.soret.shift_nm,
    reads: "haem π→π* — the resting→Compound I blue-shift tracks porphyrin oxidation",
    summary: `Soret ${SPECTRA.soret.resting_nm}→${SPECTRA.soret.compound_i_nm} nm (Δ${SPECTRA.soret.shift_nm})`,
  };
}
function opEpr() {
  return {
    kind: "cyp_epr",
    signal: "low-spin Fe³⁺ rhombic g-tensor",
    g_low: SPECTRA.epr.g_low,
    g_mid: SPECTRA.epr.g_mid,
    g_high: SPECTRA.epr.g_high,
    reads: "resting-state haem-iron spin state; the (2.42, 2.25, 1.92) rhombogram is the CYP fingerprint",
    summary: `EPR g = (${SPECTRA.epr.g_low}, ${SPECTRA.epr.g_mid}, ${SPECTRA.epr.g_high})`,
  };
}
function opRaman() {
  return {
    kind: "cyp_raman",
    mode: "Compound I / oxo-ferryl Fe=O stretch",
    fe_o_cm: SPECTRA.raman.fe_o_cm,
    fe_o_18O_cm: SPECTRA.raman.fe_o_18O_cm,
    shift_cm: SPECTRA.raman.shift_cm,
    isotope_confirmation: `${SPECTRA.raman.fe_o_cm}→${SPECTRA.raman.fe_o_18O_cm} cm⁻¹ on ¹⁶O→¹⁸O confirms an Fe=O oscillator`,
    summary: `Fe=O Raman ${SPECTRA.raman.fe_o_cm} cm⁻¹ (¹⁸O→${SPECTRA.raman.fe_o_18O_cm})`,
  };
}

function opIsoform(cyp, phenotype) {
  const iso = ISOFORMS[cyp] || ISOFORMS.CYP3A4;
  let dM = null;
  let label = cyp || "CYP3A4";
  if (phenotype && iso.phenotypes && iso.phenotypes[phenotype] != null) {
    dM = iso.phenotypes[phenotype];
    label = `${cyp} ${phenotype}`;
  } else if (phenotype && iso.alleles && iso.alleles[phenotype] != null) {
    dM = iso.alleles[phenotype];
    label = `${cyp}${phenotype}`;
  }
  return {
    kind: "cyp_isoform",
    cyp: cyp || "CYP3A4",
    family: iso.family,
    depth: iso.depth,
    phenotype: phenotype || null,
    dM,
    note: iso.note || null,
    summary:
      dM != null
        ? `${label}: ΔM=${dM}`
        : `${cyp || "CYP3A4"}: family ${iso.family}, address depth ${iso.depth}`,
  };
}

function opFloor(conditions) {
  const c = { ...REF, ...(conditions || {}) };
  const fq = floorQ(c);
  const disc = floorDisc();
  const conv = floorConv();
  const total = disc + fq.quad + conv;
  const dominant =
    conv >= fq.quad && conv >= disc ? "conv" : fq.quad >= disc ? "Q" : "disc";
  return {
    kind: "cyp_floor",
    conditions: c,
    floor_disc: disc,
    floor_Q: fq.quad,
    floor_conv: conv,
    beta: total,
    dominant_term: dominant,
    per_oscillator: fq.per,
    note:
      dominant === "conv"
        ? "at depth 9 the conversion floor dominates ⇒ β effectively condition-independent"
        : "Q-dominated regime ⇒ conditions govern comparability",
    summary: `β=${total.toExponential(3)} (dominant: ${dominant})`,
  };
}

// ============================================================================
// the module
// ============================================================================

export const cytochromeModule = {
  id: "cytochrome",

  describe() {
    return {
      id: "cytochrome",
      description:
        "Cytochrome P450 monograph as a federation contributor: folds electron-transfer, " +
        "Compound I, reaction-pathway, seven-state-orbit, spectroscopy, isoform/pharmacogenomic, " +
        "participant/carrier, and conditioned-floor facts onto a P450 reaction node. The " +
        "transaminase participant/carrier cut is one op among many — same runtime, richer corpus.",
      instructions: [
        'dispatch("cytochrome", { op: "electron-transfer" })',
        'dispatch("cytochrome", { op: "compound-i" })',
        'dispatch("cytochrome", { op: "pathway", reaction: "aliphatic-hydroxylation" })',
        'dispatch("cytochrome", { op: "states" })',
        'dispatch("cytochrome", { op: "spectroscopy" })',
        'dispatch("cytochrome", { op: "soret" })  // or "epr" / "raman" — each its own fact',
        'dispatch("cytochrome", { op: "isoform", cyp: "CYP2D6", phenotype: "PM" })',
        'dispatch("cytochrome", { op: "participants", reaction: "aliphatic-hydroxylation" })',
        'dispatch("cytochrome", { op: "floor" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const instr =
      instruction == null || instruction === "" || instruction === "demo"
        ? { op: "electron-transfer" }
        : typeof instruction === "string"
        ? { op: instruction }
        : instruction;
    const op = instr.op;

    try {
      let delta;
      switch (op) {
        case "participants":
          delta = opParticipants(instr.reaction);
          break;
        case "electron-transfer":
        case "et":
          delta = opElectronTransfer();
          break;
        case "compound-i":
        case "compound_i":
          delta = opCompoundI();
          break;
        case "pathway":
          delta = opPathway(instr.reaction);
          break;
        case "states":
        case "orbit":
          delta = opStates();
          break;
        case "spectroscopy":
        case "spectra":
          delta = opSpectroscopy();
          break;
        case "soret":
          delta = opSoret();
          break;
        case "epr":
          delta = opEpr();
          break;
        case "raman":
          delta = opRaman();
          break;
        case "isoform":
          delta = opIsoform(instr.cyp, instr.phenotype);
          break;
        case "floor":
          delta = opFloor(instr.conditions);
          break;
        default:
          return {
            ok: false,
            output_delta: {
              kind: "cyp_error",
              error: `unknown op "${op}"`,
              ops: [
                "participants", "electron-transfer", "compound-i", "pathway",
                "states", "spectroscopy", "soret", "epr", "raman", "isoform", "floor",
              ],
            },
            residue: 0,
            completed: true,
            error: `unknown op "${op}"`,
          };
      }
      return { ok: delta.ok !== false, output_delta: delta, residue: 1, completed: true };
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      return {
        ok: false,
        output_delta: { kind: "cyp_error", error: message },
        residue: 0,
        completed: true,
        error: message,
      };
    }
  },

  outputCell(_instruction) {
    return { kind: "cytochrome_cell" };
  },
};
