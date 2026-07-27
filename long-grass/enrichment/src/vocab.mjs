/* ============================================================================
 * Shared vocabulary — the IRI schema of BOTH pipelines.
 *
 * These constants mirror the nfdi4cat transaminase-kg namespaces verbatim
 * (see nfdi4cat/src/transaminase_kg/reference.py + graph.py). Enrichment
 * triples MUST reuse these exact IRIs so that a derived fact attaches to the
 * SAME subject the base graph already minted — never a parallel identity.
 *
 * The `buhera:` and `prov:` namespaces are ours: they identify the derivation
 * activity and the module that produced each fact. Everything under `ta:` /
 * `res:` is the partner pipeline's; we write onto it, we never redefine it.
 * ========================================================================== */

import N3 from "n3";
const { DataFactory } = N3;
const { namedNode, literal, quad, blankNode } = DataFactory;

// --- partner pipeline (nfdi4cat) — reused verbatim --------------------------
export const BASE_IRI = "https://w3id.org/nfdi4cat/transaminase-kg";
export const ONTOLOGY_IRI = `${BASE_IRI}/ontology`;

export const NS = {
  ta: `${ONTOLOGY_IRI}#`,
  res: `${BASE_IRI}/resource/`,
  skos: "http://www.w3.org/2004/02/skos/core#",
  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
  rdfs: "http://www.w3.org/2000/01/rdf-schema#",
  owl: "http://www.w3.org/2002/07/owl#",
  xsd: "http://www.w3.org/2001/XMLSchema#",
  dcterms: "http://purl.org/dc/terms/",
  // --- Buhera-side namespaces (ours) ---
  prov: "http://www.w3.org/ns/prov#",
  buhera: "https://buhera.dev/module/", // module identities: buhera:balance-check, buhera:lavoisier, …
  enr: `${BASE_IRI}/enrichment#`, // predicates/classes the enrichment layer mints
};

// The named graph every enrichment triple lands in. Dropping this graph
// returns the served artefact to the base pipeline's output, bit for bit.
export const ENRICHMENT_GRAPH = namedNode(`${BASE_IRI}/enrichment`);

// --- term helpers -----------------------------------------------------------
export const iri = (s) => namedNode(s);
export const ta = (local) => namedNode(NS.ta + local);
export const enr = (local) => namedNode(NS.enr + local);
export const buheraMod = (id) => namedNode(NS.buhera + id);

export const A = namedNode(NS.rdf + "type");
export const SKOS_EXACT = namedNode(NS.skos + "exactMatch");

export const xsdLit = (value, type) =>
  literal(String(value), namedNode(NS.xsd + type));
export const intLit = (n) => xsdLit(n, "integer");
export const decLit = (n) => xsdLit(n, "decimal");
export const boolLit = (b) => xsdLit(b ? "true" : "false", "boolean");
export const dateTimeLit = (isoString) => xsdLit(isoString, "dateTime");

export { namedNode, literal, quad, blankNode };
