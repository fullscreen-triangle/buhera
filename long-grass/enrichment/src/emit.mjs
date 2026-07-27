/* ============================================================================
 * Emit — the ping-pong send half.
 *
 * Collects derived facts as quads in ONE named graph (ENRICHMENT_GRAPH) and
 * serialises them back to Turtle. Detachability is the contract:
 *
 *   - every quad lands in ENRICHMENT_GRAPH, never the default graph;
 *   - the base subject IRIs are reused verbatim (facts attach, never fork);
 *   - each derived subject carries PROV: wasGeneratedBy a run activity, and
 *     the activity records the Buhera module (prov:wasAssociatedWith) that
 *     produced it.
 *
 * `DROP GRAPH <…/enrichment>` on the merged store therefore returns the
 * partner's artefact bit-for-bit. The send half writes; it never mutates a
 * base triple.
 * ========================================================================== */

import N3 from "n3";
import {
  NS,
  ENRICHMENT_GRAPH,
  enr,
  buheraMod,
  iri,
  namedNode,
  literal,
  quad,
  dateTimeLit,
} from "./vocab.mjs";

const PROV = (local) => namedNode(NS.prov + local);
const A = namedNode(NS.rdf + "type");
const RDFS_LABEL = namedNode(NS.rdfs + "label");

/**
 * Accumulates enrichment quads and their provenance.
 *
 * A single Emitter instance corresponds to one ping-pong round: one derivation
 * "batch" (a prov:Activity) under which every derived fact is generated.
 */
export class Emitter {
  /**
   * @param {object} opts
   * @param {string} opts.runId      short id for this enrichment round
   * @param {string} opts.startedAt  ISO timestamp captured by the caller
   */
  constructor({ runId, startedAt }) {
    this.store = new N3.Store();
    this.runId = runId;
    this.startedAt = startedAt;
    this.batch = iri(`${NS.res}enrichment/activity/${runId}`);
    this._activityDeclared = false;
    this._modules = new Set();
  }

  _g(s, p, o) {
    this.store.addQuad(quad(s, p, o, ENRICHMENT_GRAPH));
  }

  _ensureActivity() {
    if (this._activityDeclared) return;
    this._g(this.batch, A, PROV("Activity"));
    this._g(this.batch, PROV("startedAtTime"), dateTimeLit(this.startedAt));
    this._g(
      this.batch,
      RDFS_LABEL,
      literal(`Buhera enrichment round ${this.runId}`, "en"),
    );
    this._activityDeclared = true;
  }

  /**
   * Attach a derived fact to a base subject.
   *
   * @param {string}        subject   base-graph IRI the fact belongs to
   * @param {NamedNode}     predicate enrichment predicate (usually enr(...))
   * @param {Term}          object    N3 term (literal or namedNode)
   * @param {object}        prov
   * @param {string}        prov.module  Buhera module id that derived it
   */
  fact(subject, predicate, object, { module }) {
    this._ensureActivity();
    const subj = iri(subject);
    this._g(subj, predicate, object);
    if (module) this._modules.add(module);
  }

  /**
   * Reify a derived fact as its own node when it needs per-fact provenance
   * (evidence, the module, a numeric detail). Returns the statement node so
   * the caller can hang extra detail off it.
   */
  reifiedFact(subject, predicate, object, { module }) {
    this._ensureActivity();
    const stmt = iri(
      `${NS.res}enrichment/stmt/${this.runId}/${this.store.size}`,
    );
    const subj = iri(subject);
    // the plain assertion, in-graph
    this._g(subj, predicate, object);
    // its provenance record
    this._g(stmt, A, PROV("Entity"));
    this._g(stmt, enr("aboutSubject"), subj);
    this._g(stmt, enr("aboutPredicate"), predicate);
    this._g(stmt, PROV("value"), object);
    this._g(stmt, PROV("wasGeneratedBy"), this.batch);
    this._g(stmt, PROV("wasDerivedFrom"), subj);
    if (module) {
      this._g(stmt, PROV("wasAttributedTo"), buheraMod(module));
      this._modules.add(module);
    }
    return stmt;
  }

  /** Close the activity: end time + module associations. */
  finalize(endedAt) {
    this._ensureActivity();
    this._g(this.batch, PROV("endedAtTime"), dateTimeLit(endedAt));
    for (const m of this._modules) {
      const mod = buheraMod(m);
      this._g(mod, A, PROV("SoftwareAgent"));
      this._g(mod, RDFS_LABEL, literal(m, "en"));
      this._g(this.batch, PROV("wasAssociatedWith"), mod);
    }
  }

  /** Serialise the enrichment graph to Turtle (TriG-free: single named graph). */
  async toTurtle() {
    const writer = new N3.Writer({
      prefixes: {
        ta: NS.ta,
        res: NS.res,
        enr: NS.enr,
        skos: NS.skos,
        prov: NS.prov,
        buhera: NS.buhera,
        rdfs: NS.rdfs,
        xsd: NS.xsd,
      },
    });
    // emit as plain triples (the graph identity is carried by the file/endpoint,
    // and the caller merges into ENRICHMENT_GRAPH on the partner side).
    for (const q of this.store.getQuads(null, null, null, ENRICHMENT_GRAPH)) {
      writer.addQuad(quad(q.subject, q.predicate, q.object));
    }
    return new Promise((resolve, reject) => {
      writer.end((err, result) => (err ? reject(err) : resolve(result)));
    });
  }

  /** Serialise as TriG, preserving the named graph explicitly. */
  async toTriG() {
    const writer = new N3.Writer({
      format: "application/trig",
      prefixes: {
        ta: NS.ta,
        res: NS.res,
        enr: NS.enr,
        skos: NS.skos,
        prov: NS.prov,
        buhera: NS.buhera,
        rdfs: NS.rdfs,
        xsd: NS.xsd,
      },
    });
    writer.addQuads(this.store.getQuads(null, null, null, ENRICHMENT_GRAPH));
    return new Promise((resolve, reject) => {
      writer.end((err, result) => (err ? reject(err) : resolve(result)));
    });
  }
}
