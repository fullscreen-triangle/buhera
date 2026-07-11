/**
 * Contact Map Engine
 * Ported from run_validation.py — zero free parameters.
 * Thermodynamic data: eQuilibrator / Alberty 2003.
 * Kinetic data: BRENDA kcat (human, pH 7.4, 37C).
 * Concentrations: HMDB physiological reference.
 */

const R_GAS = 8.314e-3;  // kJ/(mol*K)
const T_PHYS = 310.0;     // K (37C)
const RT = R_GAS * T_PHYS; // 2.577 kJ/mol

// ── Pathway data ──

export function glycolysisData() {
  return {
    name: "Glycolysis",
    species: [
      { id: "Glucose", name: "D-Glucose",                mu0: -917.0,  conc: 5.0e-3,   compartment: "cytoplasm" },
      { id: "G6P",     name: "Glucose 6-phosphate",      mu0: -1318.0, conc: 0.083e-3,  compartment: "cytoplasm" },
      { id: "F6P",     name: "Fructose 6-phosphate",      mu0: -1321.0, conc: 0.016e-3,  compartment: "cytoplasm" },
      { id: "FBP",     name: "Fructose 1,6-bisphosphate", mu0: -2202.0, conc: 0.031e-3,  compartment: "cytoplasm" },
      { id: "G3P",     name: "Glyceraldehyde 3-phosphate", mu0: -1285.0, conc: 0.019e-3, compartment: "cytoplasm" },
      { id: "BPG13",   name: "1,3-Bisphosphoglycerate",   mu0: -2356.0, conc: 0.001e-3,  compartment: "cytoplasm" },
      { id: "3PG",     name: "3-Phosphoglycerate",         mu0: -1502.0, conc: 0.12e-3,   compartment: "cytoplasm" },
      { id: "2PG",     name: "2-Phosphoglycerate",         mu0: -1497.0, conc: 0.03e-3,   compartment: "cytoplasm" },
      { id: "PEP",     name: "Phosphoenolpyruvate",        mu0: -1269.0, conc: 0.023e-3,  compartment: "cytoplasm" },
      { id: "Pyruvate", name: "Pyruvate",                  mu0: -472.0,  conc: 0.051e-3,  compartment: "cytoplasm" },
    ],
    reactions: [
      { src: "Glucose", dst: "G6P",     enzyme: "Hexokinase (HK1)",           kcat: 240.0 },
      { src: "G6P",     dst: "F6P",     enzyme: "Phosphoglucose isomerase",   kcat: 1240.0 },
      { src: "F6P",     dst: "FBP",     enzyme: "Phosphofructokinase (PFK1)", kcat: 150.0 },
      { src: "FBP",     dst: "G3P",     enzyme: "Aldolase",                   kcat: 18.0 },
      { src: "G3P",     dst: "BPG13",   enzyme: "GAPDH",                      kcat: 130.0 },
      { src: "BPG13",   dst: "3PG",     enzyme: "Phosphoglycerate kinase",    kcat: 370.0 },
      { src: "3PG",     dst: "2PG",     enzyme: "Phosphoglycerate mutase",    kcat: 795.0 },
      { src: "2PG",     dst: "PEP",     enzyme: "Enolase",                    kcat: 80.0 },
      { src: "PEP",     dst: "Pyruvate", enzyme: "Pyruvate kinase (PKM2)",    kcat: 550.0 },
    ],
  };
}

export function tcaCycleData() {
  return {
    name: "TCA Cycle",
    species: [
      { id: "AcCoA",   name: "Acetyl-CoA",         mu0: -374.0,  conc: 0.06e-3,  compartment: "mito_matrix" },
      { id: "Citrate", name: "Citrate",             mu0: -1166.0, conc: 0.44e-3,  compartment: "mito_matrix" },
      { id: "Isocit",  name: "Isocitrate",          mu0: -1160.0, conc: 0.04e-3,  compartment: "mito_matrix" },
      { id: "aKG",     name: "alpha-Ketoglutarate", mu0: -798.0,  conc: 0.03e-3,  compartment: "mito_matrix" },
      { id: "SucCoA",  name: "Succinyl-CoA",        mu0: -509.0,  conc: 0.05e-3,  compartment: "mito_matrix" },
      { id: "Succ",    name: "Succinate",            mu0: -690.0,  conc: 0.30e-3,  compartment: "mito_matrix" },
      { id: "Fum",     name: "Fumarate",             mu0: -604.0,  conc: 0.03e-3,  compartment: "mito_matrix" },
      { id: "Malate",  name: "Malate",               mu0: -842.0,  conc: 0.22e-3,  compartment: "mito_matrix" },
      { id: "OAA",     name: "Oxaloacetate",         mu0: -794.0,  conc: 0.011e-3, compartment: "mito_matrix" },
    ],
    reactions: [
      { src: "AcCoA",   dst: "Citrate", enzyme: "Citrate synthase",         kcat: 167.0 },
      { src: "Citrate", dst: "Isocit",  enzyme: "Aconitase",                kcat: 30.0 },
      { src: "Isocit",  dst: "aKG",     enzyme: "Isocitrate dehydrogenase", kcat: 28.0 },
      { src: "aKG",     dst: "SucCoA",  enzyme: "alpha-KG dehydrogenase",   kcat: 50.0 },
      { src: "SucCoA",  dst: "Succ",    enzyme: "Succinyl-CoA ligase",      kcat: 22.0 },
      { src: "Succ",    dst: "Fum",     enzyme: "Succinate dehydrogenase",   kcat: 19.0 },
      { src: "Fum",     dst: "Malate",  enzyme: "Fumarase",                 kcat: 800.0 },
      { src: "Malate",  dst: "OAA",     enzyme: "Malate dehydrogenase",     kcat: 350.0 },
      { src: "OAA",     dst: "AcCoA",   enzyme: "OAA to AcCoA (cycle)",     kcat: 100.0 },
    ],
  };
}

export function oxphosData() {
  return {
    name: "Oxidative Phosphorylation",
    species: [
      { id: "NADH", name: "NADH",               mu0: -32.0,   conc: 0.10e-3,  compartment: "mito_matrix" },
      { id: "CoQ",  name: "Ubiquinone (CoQ10)",  mu0: -36.0,   conc: 2.0e-3,   compartment: "mito_IMM" },
      { id: "CytC", name: "Cytochrome c",        mu0: -13.0,   conc: 0.50e-3,  compartment: "mito_IMS" },
      { id: "O2",   name: "Molecular oxygen",    mu0: 0.0,     conc: 0.025e-3, compartment: "mito_matrix" },
      { id: "H2O",  name: "Water",               mu0: -237.0,  conc: 55.5,     compartment: "mito_matrix" },
      { id: "ADP",  name: "ADP",                 mu0: -1906.0, conc: 1.3e-3,   compartment: "mito_matrix" },
      { id: "Pi",   name: "Inorganic phosphate", mu0: -1059.0, conc: 10.0e-3,  compartment: "mito_matrix" },
      { id: "ATP",  name: "ATP",                 mu0: -2768.0, conc: 3.2e-3,   compartment: "mito_matrix" },
    ],
    reactions: [
      { src: "NADH", dst: "CoQ",  enzyme: "Complex I",           kcat: 500.0 },
      { src: "CoQ",  dst: "CytC", enzyme: "Complex III",         kcat: 250.0 },
      { src: "CytC", dst: "O2",   enzyme: "Complex IV",          kcat: 350.0 },
      { src: "O2",   dst: "H2O",  enzyme: "Water formation",     kcat: 350.0 },
      { src: "ADP",  dst: "ATP",  enzyme: "ATP synthase",        kcat: 100.0 },
      { src: "NADH", dst: "ADP",  enzyme: "PMF coupling",        kcat: 80.0 },
      { src: "ATP",  dst: "Pi",   enzyme: "ATP hydrolysis",      kcat: 10.0 },
    ],
  };
}

export function egfrMapkData() {
  return {
    name: "EGFR/MAPK Signalling",
    species: [
      { id: "EGF",  name: "EGF",       mu0: -50.0, conc: 1.0e-9,  compartment: "extracellular" },
      { id: "EGFR", name: "EGFR",      mu0: -40.0, conc: 1.0e-7,  compartment: "membrane" },
      { id: "GRB2", name: "GRB2",      mu0: -30.0, conc: 0.5e-6,  compartment: "cytoplasm" },
      { id: "SOS",  name: "SOS1",      mu0: -25.0, conc: 0.1e-6,  compartment: "cytoplasm" },
      { id: "RAS",  name: "KRAS",      mu0: -20.0, conc: 0.5e-6,  compartment: "membrane" },
      { id: "RAF",  name: "BRAF",      mu0: -15.0, conc: 0.3e-6,  compartment: "cytoplasm" },
      { id: "MEK",  name: "MEK1/2",    mu0: -12.0, conc: 1.2e-6,  compartment: "cytoplasm" },
      { id: "ERK",  name: "ERK1/2",    mu0: -10.0, conc: 1.0e-6,  compartment: "cytoplasm" },
      { id: "MYC",  name: "c-MYC",     mu0: -8.0,  conc: 0.01e-6, compartment: "nucleus" },
      { id: "CycD", name: "Cyclin D1", mu0: -5.0,  conc: 0.05e-6, compartment: "nucleus" },
    ],
    reactions: [
      { src: "EGF",  dst: "EGFR", enzyme: "EGF-EGFR binding",         kcat: 1.0e6 },
      { src: "EGFR", dst: "GRB2", enzyme: "EGFR autophosphorylation", kcat: 10.0 },
      { src: "GRB2", dst: "SOS",  enzyme: "GRB2-SOS recruitment",     kcat: 5.0 },
      { src: "SOS",  dst: "RAS",  enzyme: "SOS RAS-GTP exchange",     kcat: 0.5 },
      { src: "RAS",  dst: "RAF",  enzyme: "RAS RAF activation",       kcat: 2.0 },
      { src: "RAF",  dst: "MEK",  enzyme: "RAF MEK phosphorylation",  kcat: 8.0 },
      { src: "MEK",  dst: "ERK",  enzyme: "MEK ERK phosphorylation",  kcat: 15.0 },
      { src: "ERK",  dst: "MYC",  enzyme: "ERK MYC stabilisation",    kcat: 3.0 },
      { src: "MYC",  dst: "CycD", enzyme: "MYC CyclinD transcription", kcat: 0.8 },
      { src: "ERK",  dst: "EGFR", enzyme: "ERK EGFR feedback",        kcat: 1.0 },
    ],
  };
}

export const PATHWAYS = { glycolysis: glycolysisData, tca: tcaCycleData, oxphos: oxphosData, egfr: egfrMapkData };
export const DISEASE_MODELS = {
  hk1:      { pathway: "glycolysis", edge: 0, alpha: 0.1,  label: "HK1 deficiency" },
  idh2:     { pathway: "tca",        edge: 2, alpha: 0.15, label: "IDH2 R172K" },
  rotenone: { pathway: "oxphos",     edge: 0, alpha: 0.3,  label: "Complex I (rotenone)" },
  kras:     { pathway: "egfr",       edge: 3, alpha: 5.0,  label: "KRAS G12V" },
};

// ── Core engine ──

function deepClone(obj) { return JSON.parse(JSON.stringify(obj)); }

export function computeChemicalPotentials(species) {
  for (const s of species) s.mu = s.mu0 + RT * Math.log(s.conc);
  return species;
}

export function computeConductances(species, reactions) {
  const sp = Object.fromEntries(species.map(s => [s.id, s]));
  for (const r of reactions) r.conductance = r.kcat * sp[r.src].conc / RT;
  return reactions;
}

export function computeFluxes(species, reactions) {
  const sp = Object.fromEntries(species.map(s => [s.id, s]));
  for (const r of reactions) r.flux = r.conductance * (sp[r.src].mu - sp[r.dst].mu);
  return reactions;
}

export function computeSentropy(species, reactions) {
  const neighbours = Object.fromEntries(species.map(s => [s.id, []]));
  for (const r of reactions) {
    neighbours[r.src].push(r);
    neighbours[r.dst].push({ src: r.dst, dst: r.src, flux: -r.flux, conductance: r.conductance });
  }

  const fluxTotals = {}, condTotals = {};
  for (const s of species) {
    fluxTotals[s.id] = neighbours[s.id].reduce((a, r) => a + Math.abs(r.flux), 0);
    condTotals[s.id] = neighbours[s.id].reduce((a, r) => a + r.conductance, 0);
  }

  const Fmax = Math.max(...Object.values(fluxTotals)) || 1;
  const Cmax = Math.max(...Object.values(condTotals)) || 1;
  const mus = species.map(s => s.mu);
  const muMin = Math.min(...mus), muMax = Math.max(...mus);
  const muRange = muMax !== muMin ? muMax - muMin : 1;

  for (const s of species) {
    s.Sk = fluxTotals[s.id] / Fmax;
    s.St = condTotals[s.id] / Cmax;
    s.Se = (s.mu - muMin) / muRange;
  }
  return species;
}

export function sentropyDistance(s1, s2) {
  return Math.sqrt((s1.Sk - s2.Sk) ** 2 + (s1.St - s2.St) ** 2 + (s1.Se - s2.Se) ** 2);
}

export function computeContactMap(speciesIn, reactionsIn) {
  const species = deepClone(speciesIn);
  const reactions = deepClone(reactionsIn);

  computeChemicalPotentials(species);
  computeConductances(species, reactions);
  computeFluxes(species, reactions);
  computeSentropy(species, reactions);

  const sp = Object.fromEntries(species.map(s => [s.id, s]));
  const edges = reactions.map(r => {
    const src = sp[r.src], dst = sp[r.dst];
    return {
      src: r.src, dst: r.dst, enzyme: r.enzyme,
      cost: sentropyDistance(src, dst),
      conductance: r.conductance, flux: r.flux,
      srcS: [src.Sk, src.St, src.Se],
      dstS: [dst.Sk, dst.St, dst.Se],
    };
  });

  return { species, reactions, edges };
}

// ── Spearman rank correlation (no scipy needed) ──

function rankArray(arr) {
  const indexed = arr.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
    const avgRank = (i + j - 1) / 2 + 1;
    for (let k = i; k < j; k++) ranks[indexed[k].i] = avgRank;
    i = j;
  }
  return ranks;
}

function spearmanR(a, b) {
  const ra = rankArray(a), rb = rankArray(b);
  const n = a.length;
  const meanA = ra.reduce((s, v) => s + v, 0) / n;
  const meanB = rb.reduce((s, v) => s + v, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = ra[i] - meanA, db = rb[i] - meanB;
    num += da * db; denA += da * da; denB += db * db;
  }
  return denA && denB ? num / Math.sqrt(denA * denB) : 0;
}

export function tripleCoherence(species) {
  const Sk = species.map(s => s.Sk), St = species.map(s => s.St), Se = species.map(s => s.Se);
  const rhoKT = spearmanR(Sk, St), rhoTE = spearmanR(St, Se), rhoKE = spearmanR(Sk, Se);
  return { R: (rhoKT + rhoTE + rhoKE) / 3, rhoKT, rhoTE, rhoKE };
}

export function fluxVisibility(rxnHealthy, rxnPerturbed) {
  const totalG = rxnHealthy.reduce((a, r) => a + r.conductance, 0);
  if (!totalG) return 0;
  let logV = 0;
  for (let i = 0; i < rxnHealthy.length; i++) {
    const w = rxnHealthy[i].conductance / totalG;
    const Jh = Math.abs(rxnHealthy[i].flux), Jp = Math.abs(rxnPerturbed[i].flux);
    if (!Jh || !Jp) { logV += w * Math.log(1e-12); continue; }
    logV += w * Math.log(Math.max(Math.min(Jh, Jp) / Math.max(Jh, Jp), 1e-12));
  }
  return Math.exp(logV);
}

export function backwardNavigation(species, reactions, targetId) {
  const incoming = Object.fromEntries(species.map(s => [s.id, []]));
  const outgoing = Object.fromEntries(species.map(s => [s.id, []]));
  for (const r of reactions) { outgoing[r.src].push(r); incoming[r.dst].push(r); }

  const path = [targetId], visited = new Set([targetId]);
  let current = targetId;
  for (let step = 0; step < species.length; step++) {
    const cands = [];
    for (const r of incoming[current]) if (!visited.has(r.src)) cands.push([r.conductance, r.src]);
    for (const r of outgoing[current]) if (!visited.has(r.dst)) cands.push([r.conductance, r.dst]);
    if (!cands.length) break;
    cands.sort((a, b) => b[0] - a[0]);
    const next = cands[0][1];
    path.push(next); visited.add(next); current = next;
  }
  return path;
}

export function applyPerturbation(speciesIn, reactionsIn, edgeIdx, alpha) {
  const species = deepClone(speciesIn);
  const reactions = deepClone(reactionsIn);
  computeChemicalPotentials(species);
  computeConductances(species, reactions);
  reactions[edgeIdx].conductance *= alpha;
  computeFluxes(species, reactions);
  computeSentropy(species, reactions);
  return { species, reactions };
}

export function analysePathway(pathwayKey) {
  const data = PATHWAYS[pathwayKey]();
  const { species, reactions, edges } = computeContactMap(data.species, data.reactions);
  const tc = tripleCoherence(species);

  const filtration = [...edges].sort((a, b) => a.cost - b.cost);

  return {
    name: data.name,
    species, reactions, edges,
    tripleCoherence: tc,
    filtration,
  };
}

export function analyseDisease(diseaseKey) {
  const dm = DISEASE_MODELS[diseaseKey];
  const data = PATHWAYS[dm.pathway]();
  const healthy = computeContactMap(data.species, data.reactions);
  const perturbed = applyPerturbation(data.species, data.reactions, dm.edge, dm.alpha);
  const V = fluxVisibility(healthy.reactions, perturbed.reactions);
  const tcHealthy = tripleCoherence(healthy.species);
  const tcPerturbed = tripleCoherence(perturbed.species);
  const targetId = healthy.species[healthy.species.length - 1].id;
  const nav = backwardNavigation(perturbed.species, perturbed.reactions, targetId);

  const shifts = healthy.species.map((hs, i) => {
    const ds = perturbed.species[i];
    const shift = Math.sqrt((hs.Sk - ds.Sk) ** 2 + (hs.St - ds.St) ** 2 + (hs.Se - ds.Se) ** 2);
    return { id: hs.id, shift, healthyS: [hs.Sk, hs.St, hs.Se], diseasedS: [ds.Sk, ds.St, ds.Se] };
  });

  return {
    label: dm.label,
    pathway: data.name,
    alpha: dm.alpha,
    edgeIdx: dm.edge,
    perturbedEdge: `${data.reactions[dm.edge].src} -> ${data.reactions[dm.edge].dst}`,
    V,
    R_healthy: tcHealthy.R,
    R_diseased: tcPerturbed.R,
    navigation: nav,
    shifts,
    mostAffected: shifts.reduce((a, b) => b.shift > a.shift ? b : a).id,
  };
}
