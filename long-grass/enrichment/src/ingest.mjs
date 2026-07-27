/* ============================================================================
 * Ingest — the ping-pong receive half.
 *
 * Reads the partner pipeline's emitted RDF graph (Turtle) and builds a small
 * queryable model of exactly what the WIRE carries: species (with their ChEBI
 * accession, lifted from skos:exactMatch) and reactions (with substrate /
 * product / acceptor / donor roles).
 *
 * It deliberately reads ONLY the served graph — no molecular formulas, no
 * charges, no access to the partner's source code. Whatever the base graph
 * does not state, Buhera must DERIVE (that is the whole point). Formulas, for
 * instance, are absent from the wire and are resolved downstream from the
 * ChEBI IRI, not smuggled in from the partner's internals.
 * ========================================================================== */

import { readFile } from "node:fs/promises";
import N3 from "n3";
import { NS } from "./vocab.mjs";

const CHEBI_PREFIX = "http://purl.obolibrary.org/obo/CHEBI_";

/** Parse a Turtle file into an N3.Store. */
export async function loadTurtle(path) {
  const text = await readFile(path, "utf8");
  const parser = new N3.Parser({ format: "text/turtle" });
  const store = new N3.Store();
  store.addQuads(parser.parse(text));
  return store;
}

const T = (local) => NS.ta + local;
const S = (local) => NS.skos + local;
const RDF_TYPE = NS.rdf + "type";
const RDFS_LABEL = NS.rdfs + "label";

function objectsOf(store, subject, predicate) {
  return store.getQuads(subject, predicate, null, null).map((q) => q.object);
}
function oneObject(store, subject, predicate) {
  const os = objectsOf(store, subject, predicate);
  return os.length ? os[0] : null;
}

/** Lift the ChEBI accession (e.g. "16977") from a species' skos:exactMatch. */
function chebiAccession(store, speciesIri) {
  for (const o of objectsOf(store, speciesIri, S("exactMatch"))) {
    if (o.value.startsWith(CHEBI_PREFIX)) {
      return o.value.slice(CHEBI_PREFIX.length);
    }
  }
  return null;
}

/**
 * Build the ingest model from a parsed store.
 *
 * @returns {{
 *   species: Map<string, {iri, label, chebi, isCofactor}>,
 *   reactions: Array<{iri, key, label, ec, substrates: string[], products: string[],
 *                     acceptor: string|null, donor: string|null, cofactor: string|null}>,
 * }}
 */
export function buildModel(store) {
  // --- species (ChemicalSpecies + Cofactor) ---
  const species = new Map();
  const speciesTypes = [T("ChemicalSpecies"), T("Cofactor")];
  for (const typeIri of speciesTypes) {
    for (const q of store.getQuads(null, RDF_TYPE, typeIri, null)) {
      const iri = q.subject.value;
      const labelNode = oneObject(store, q.subject, RDFS_LABEL);
      species.set(iri, {
        iri,
        label: labelNode ? labelNode.value : iri,
        chebi: chebiAccession(store, q.subject),
        isCofactor: typeIri === T("Cofactor"),
      });
    }
  }

  // --- enzymes: map enzyme IRI -> cofactor IRI, for reaction lookups ---
  const enzymeCofactor = new Map();
  for (const q of store.getQuads(null, RDF_TYPE, T("Enzyme"), null)) {
    const cof = oneObject(store, q.subject, T("hasCofactor"));
    if (cof) enzymeCofactor.set(q.subject.value, cof.value);
  }

  // --- reactions (Transamination) ---
  const reactions = [];
  for (const q of store.getQuads(null, RDF_TYPE, T("Transamination"), null)) {
    const subj = q.subject;
    const iri = subj.value;
    const labelNode = oneObject(store, subj, RDFS_LABEL);
    const catalyzedBy = oneObject(store, subj, T("catalyzedBy"));
    const acceptor = oneObject(store, subj, T("hasAminoAcceptor"));
    const donor = oneObject(store, subj, T("hasAminoDonor"));
    const enzymeIri = catalyzedBy ? catalyzedBy.value : null;

    let ec = null;
    if (enzymeIri) {
      const ecNode = oneObject(store, catalyzedBy, T("ecNumber"));
      ec = ecNode ? ecNode.value : null;
    }

    reactions.push({
      iri,
      key: iri.slice(iri.lastIndexOf("/") + 1),
      label: labelNode ? labelNode.value : iri,
      ec,
      enzyme: enzymeIri,
      substrates: objectsOf(store, subj, T("hasSubstrate")).map((o) => o.value),
      products: objectsOf(store, subj, T("hasProduct")).map((o) => o.value),
      acceptor: acceptor ? acceptor.value : null,
      donor: donor ? donor.value : null,
      cofactor: enzymeIri ? enzymeCofactor.get(enzymeIri) ?? null : null,
    });
  }

  return { species, reactions };
}
