/* /data-modeling — an interactive report on Buhera OS's data-modeling
 * primitive: shape (LinkML and the shape-language family) composed with a
 * floor/residue layer that certifies admissibility, which no shape language
 * can. Companion to the /protein-modelling report; same house style
 * (Section/Fig wrappers, numbered sections, dark theme), but every figure
 * here is a live d3 chart or interactive diagram rather than a rendered
 * screenshot — built from the data-modeling-capability.tex paper.
 */
import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";

const CapabilityVenn = dynamic(() => import("@/components/datamodel/CapabilityVenn"), { ssr: false });
const ContactGraphFloor = dynamic(() => import("@/components/datamodel/ContactGraphFloor"), { ssr: false });
const SeparationPair = dynamic(() => import("@/components/datamodel/SeparationPair"), { ssr: false });
const CompositionTree = dynamic(() => import("@/components/datamodel/CompositionTree"), { ssr: false });
const VerdictDecisionTree = dynamic(() => import("@/components/datamodel/VerdictDecisionTree"), { ssr: false });
const LadderComposition = dynamic(() => import("@/components/datamodel/LadderComposition"), { ssr: false });

function Section({ id, n, title, children }) {
  return (
    <section id={id} className="mt-10 scroll-mt-20">
      <h2 className="text-2xl font-semibold text-white mb-3 border-b border-gray-800 pb-1">
        <span className="text-gray-600 font-mono text-lg mr-2">{n}</span>
        {title}
      </h2>
      <div className="space-y-4 text-gray-300 leading-relaxed">{children}</div>
    </section>
  );
}

function Fig({ caption, children }) {
  return (
    <figure className="my-5 border border-gray-800 rounded bg-gray-950 p-4">
      <div className="overflow-x-auto flex justify-center">{children}</div>
      <figcaption className="mt-3 text-sm text-gray-500 leading-relaxed">{caption}</figcaption>
    </figure>
  );
}

function Cite({ n }) {
  return (
    <a href="#references" className="text-blue-400 hover:text-blue-300 no-underline text-[0.85em] align-super">
      [{n}]
    </a>
  );
}

const REFERENCES = [
  { key: "linkml", text: "LinkML: Linked Data Modeling Language. https://linkml.io/linkml/ — schemas as classes, slots, types, and enums, compiled to JSON Schema, SHACL, OWL, SQL DDL, and typed code." },
  { key: "masamune", text: "Masamune: A Verdict-Carrying Converter and Plan Language for Chemical Structure Representations. honjo-masamune/docs/masamune-representation-converter — the capability-set primitive Capset(S) ⊆ Feat this report generalizes." },
  { key: "ladder", text: "Catalytic Ladder Propagation: An Admissibility Architecture for Catalysis Knowledge Graphs. levinthal/nfdi4cat/catalytic-ladder-propagation — the floor, the ladder, and Theorem 6.4 (retrieval cannot express admissibility)." },
  { key: "trace", text: "The Trace Calculus: Residue, Fixed Points, and Sufficiency in Contact Graphs. musande/epistemology/causality/trace-calculus-operations." },
  { key: "occupation", text: "The Propagation of Occupation. musande/epistemology/occupation-propagation — the floor theorem and non-enumerability underlying every contact-graph result cited here." },
  { key: "dmc", text: "Shape Is Not Admissibility: A Capability-Theoretic Foundation for Data Modeling in Buhera OS. long-grass/docs/data-modeling-capability — the formal paper this page renders interactively." },
];
const R = Object.fromEntries(REFERENCES.map((r, i) => [r.key, i + 1]));

export default function DataModeling() {
  return (
    <>
      <Head>
        <title>Shape is not admissibility · data modeling · buhera</title>
        <meta
          name="description"
          content="An interactive report on Buhera OS's data-modeling primitive: shape (LinkML and the shape-language family) composed with a floor/residue layer that certifies admissibility."
        />
      </Head>

      <div className="min-h-screen bg-black text-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-10">
          <nav className="mb-8 text-sm flex items-center justify-between">
            <Link href="/tutorials" className="text-blue-400 hover:text-blue-300">← tutorials</Link>
            <Link href="/" className="text-blue-400 hover:text-blue-300">terminal →</Link>
          </nav>

          <header className="mb-8">
            <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">
              Buhera OS · data modeling report
            </p>
            <h1 className="text-4xl font-bold text-white leading-tight">
              Shape is <span className="text-red-400">not</span> admissibility
            </h1>
            <p className="mt-4 text-gray-400 leading-relaxed">
              LinkML, JSON Schema, SHACL, and OWL answer &quot;what does a record
              of this kind look like.&quot; They answer it well, and adopting one
              of them for a given surface — Chem-DCAT-AP for dataset-catalog
              interoperability, say — costs Buhera OS nothing. This report works
              through the other question none of them answer: given a schema
              that is complete and honest, does an agent holding it know which{" "}
              <em>questions</em> are actually answerable against the data? Every
              diagram below is live — drag, click, or slide it.
            </p>
          </header>

          <div className="border border-gray-800 rounded p-5 bg-gray-950">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-2">Abstract</h2>
            <p className="text-gray-300 text-sm leading-relaxed">
              We generalize a capability-declaration primitive already proven for
              chemical file-format conversion — a declared feature set
              Capset(S) ⊆ Feat with statically decidable containment
              <Cite n={R.masamune} /> — from file formats to data schemas of any
              kind, and show LinkML and the wider shape-language family
              instantiate it faithfully: a schema <em>is</em> a capability
              declaration. We then generalize the separation result already
              proven for SPARQL and SHACL over RDF triples
              <Cite n={R.ladder} /> to the entire shape-language family: shape-
              certification and admissibility-certification are provably
              independent, because admissibility is a global floor over a
              data source&apos;s contact structure, and every shape language&apos;s
              declarable features are local by construction. Buhera OS&apos;s
              data-modeling primitive is the composition of the two — adopt
              whatever shape language is sufficient, unmodified, and certify
              admissibility underneath it with the floor/residue layer that no
              shape language provides.
            </p>
          </div>

          {/* 1 */}
          <Section id="capability" n="1" title="A schema is a capability declaration">
            <p>
              A data source declares which features of a shared alphabet Feat it
              actually populates — its capability set Capset(S) ⊆ Feat. A
              consumer&apos;s request Req ⊆ Feat is satisfiable exactly when
              Req ⊆ Capset(S), decidable in O(|Feat|) time without reading a
              single record <Cite n={R.masamune} />. This was proven for
              chemical file formats (SMILES, molfiles, InChI); nothing in the
              definition is specific to chemistry.
            </p>
            <Fig caption="Drag the sliders: Req grows or drifts away from Capset(S). Containment — and therefore satisfiability — is a set relation checkable before any record is read.">
              <CapabilityVenn />
            </Fig>
            <p>
              LinkML schemas are exactly this: a class&apos;s required slots are
              populated on every valid instance, so they lie in every instance&apos;s
              realized capability set; optional slots contribute an
              instance-dependent Capset. JSON Schema, SHACL, and OWL follow the
              same correspondence through their own generator round-trips
              <Cite n={R.linkml} />. Nothing is lost by treating any of them this
              way — this is the sense in which adopting one for a surface Buhera
              must expose is a free, correct engineering choice.
            </p>
          </Section>

          {/* 2 */}
          <Section id="floor" n="2" title="The floor: a property no single record carries">
            <p>
              A data source&apos;s records induce a contact graph: items are
              vertices, weighted edges record relations between them, and a
              distinguished medium vertex stands for the ambient population.
              Admissibility of a question — is there an accountable path from
              v₀ to a target x* — depends on the separation cost&apos;s{" "}
              <strong className="text-white">floor</strong>, the minimum-weight
              cut over the whole graph, not on any single edge or field.
            </p>
            <Fig caption="Drag any vertex. The dashed red edge is the minimum-weight cut touching the medium — the floor β. It is read off the whole graph's structure, never off a single vertex's declared attributes.">
              <ContactGraphFloor />
            </Fig>
          </Section>

          {/* 3 */}
          <Section id="separation" n="3" title="The separation theorem">
            <p>
              This is the paper&apos;s central result, generalized from a proof
              already given for SPARQL and SHACL specifically
              <Cite n={R.ladder} />: two data sources can have identical
              declared capability sets, identical schema-conformant records, no
              discernible difference at the level of any single record — and
              still disagree on whether a given question is admissible, because
              the floor is a global minimum no local pattern-match can see.
            </p>
            <Fig caption="Same shape, same Capset(S), same fields on every record — only the weight on one peripheral contact differs, and that alone flips admissibility of v0 ⇝ x*. Adding attributes, constraints, or slots to Feat does not close this gap (Corollary: no enrichment repairs this) — the floor is not a feature any local alphabet can name.">
              <SeparationPair />
            </Fig>
          </Section>

          {/* 4 */}
          <Section id="composition" n="4" title="Composition: Buhera's data layer">
            <p>
              The consequence is constructive, not critical. A composed data
              layer is a shape source — adopted, extended, or invented, whatever
              a surface requires — paired with the contact graph its records
              induce. Neither half is derivable from the other: shape doesn&apos;t
              give you the floor, and the floor doesn&apos;t retain per-field typing.
              Both are necessary; neither alone is Buhera&apos;s data-modeling
              story.
            </p>
            <Fig caption="Shape (LinkML, JSON Schema, SHACL, OWL — all instances of the same primitive) answers 'what does this record look like.' The floor/residue layer answers 'which questions can this data actually answer' — role, direction, and admissibility, computed rather than curated.">
              <CompositionTree />
            </Fig>
          </Section>

          {/* 5 */}
          <Section id="verdicts" n="5" title="The verdict algebra">
            <p>
              A composed data layer reports outcomes as one of six mutually
              exclusive verdicts, never conflated into a single success/failure
              bit <Cite n={R.ladder} />. Only <code className="text-emerald-400">answer</code>{" "}
              carries a payload (non-degeneracy) — every other outcome is a
              distinct, named refusal with its own diagnosis, exactly the
              distinction the existing federated front end already renders live
              against real biocatalysis questions.
            </p>
            <Fig caption="Click a step's 'no' or 'yes' branch, or click a verdict directly, to see where each of the six outcomes sits in the decision path and what it means.">
              <VerdictDecisionTree />
            </Fig>
          </Section>

          {/* 6 */}
          <Section id="ladder" n="6" title="Composing evidence: the ladder">
            <p>
              Within the floor/residue layer, individual rungs of evidence
              compose multiplicatively: pow(Ladder) = 1 − Π(1 − powᵢ). Repeated
              identical rungs saturate — each additional rung of the same
              strength buys strictly less than the last — and the composite is
              most sensitive to improving its <em>strongest</em> rung under
              additive sensitivity, though flat under proportional,
              headroom-scaled improvement <Cite n={R.ladder} />.
            </p>
            <Fig caption="Adjust each rung's power and watch the composite curve. The dashed grey curve is the saturation bound for repeating the mean rung power — composite power always trails strictly below it once rungs diverge in strength.">
              <LadderComposition />
            </Fig>
          </Section>

          {/* 7 */}
          <Section id="practical" n="7" title="What this means in practice">
            <ul className="list-disc pl-6 space-y-2">
              <li>
                A dataset-catalog listing, a metadata harvester, a
                FAIR-compliance check — these are shape requests, and
                Chem-DCAT-AP already serves them well. Extending it for these
                purposes adds engineering cost for no theorem-backed gain.
              </li>
              <li>
                &quot;Which enzymes have I not yet tried against this substrate,
                and would trying them tell me anything the floor doesn&apos;t
                already foreclose&quot; is an admissibility question. No shape
                schema, however extended, answers it — this is a theorem, not a
                gap in LinkML&apos;s current coverage.
              </li>
              <li>
                Buhera OS&apos;s contribution is not a fifth shape language beside
                LinkML, JSON Schema, SHACL, and OWL. It is the floor/residue
                layer every one of them is missing, composed underneath
                whichever is sufficient for a given surface.
              </li>
            </ul>
          </Section>

          {/* references */}
          <section id="references" className="mt-12 pt-6 border-t border-gray-800">
            <h2 className="text-lg font-semibold text-white mb-3">References</h2>
            <ol className="text-sm text-gray-400 space-y-2 list-decimal pl-5">
              {REFERENCES.map((r) => (
                <li key={r.key}>{r.text}</li>
              ))}
            </ol>
          </section>

          <hr className="my-8 border-gray-800" />

          <nav className="flex items-center justify-between text-sm">
            <Link href="/protein-modelling" className="text-blue-400 hover:text-blue-300">
              ← protein modelling report
            </Link>
            <Link href="/tutorials" className="text-blue-400 hover:text-blue-300">
              all tutorials →
            </Link>
          </nav>
        </div>
      </div>
    </>
  );
}
