/* /protein-modelling — a rigorous report on modelling cytochrome P450 as a
 * categorical knowledge graph that IS its own runtime, ending in a live IDE.
 *
 * The prose is the report; the embedded <P450Ide> at the end is the apparatus.
 * Every number in the Results section is one a file in the IDE reproduces when
 * you run it — the report and the runtime are the same object seen two ways.
 */

import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";

// The notebook pulls in the terminal render tree; keep it client-only so the
// report itself stays statically rendered.
const P450Notebook = dynamic(() => import("@/components/P450Notebook"), {
  ssr: false,
});

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
      <div className="overflow-x-auto">{children}</div>
      <figcaption className="mt-3 text-sm text-gray-500 leading-relaxed">
        {caption}
      </figcaption>
    </figure>
  );
}

// An inline citation marker linking to the References section. `n` is a single
// number or a comma range like "3, 4".
function Cite({ n }) {
  return (
    <a
      href="#references"
      className="text-blue-400 hover:text-blue-300 no-underline text-[0.85em] align-super"
    >
      [{n}]
    </a>
  );
}

// The reference list. On-disk grounding papers first (the reproducible sources),
// then the external literature named in the corpus. The user's own monograph
// source is deliberately not listed.
const REFERENCES = [
  {
    key: "empty-dict",
    text:
      "The Empty Dictionary for Cytochrome P450: Constant Resident State, Query Without Entries, and a Correction to the Storage Claim of Paper 2. On-disk grounding paper (Monograph Paper 16), with validation scripts and results/*.json reproducing every reported number.",
  },
  {
    key: "db-recovery",
    text:
      "P450 Database Recovery via Ternary Address Encoding: Information Capacity, Partial Address Reconstruction, and Cross-Species Interpolation. On-disk grounding paper (Monograph Paper 13), with eight validation scripts and rendered result panels.",
  },
  {
    key: "gotoh",
    text:
      "Gotoh, O. (1992). Substrate recognition sites in cytochrome P450 family 2 proteins inferred from comparative analyses of amino acid and coding nucleotide sequences. J. Biol. Chem. 267, 83–90.",
  },
  {
    key: "kyte",
    text:
      "Kyte, J. & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. J. Mol. Biol. 157, 105–132.",
  },
  {
    key: "zamyatnin",
    text:
      "Zamyatnin, A. A. (1972). Protein volume in solution. Prog. Biophys. Mol. Biol. 24, 107–123.",
  },
  {
    key: "spearman",
    text:
      "Spearman, C. (1904). The proof and measurement of association between two things. Am. J. Psychol. 15, 72–101.",
  },
  {
    key: "berman",
    text:
      "Berman, H. M. et al. (2000). The Protein Data Bank. Nucleic Acids Res. 28, 235–242.",
  },
  {
    key: "varadi",
    text:
      "Varadi, M. et al. (2022). AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space. Nucleic Acids Res. 50, D439–D444.",
  },
  {
    key: "gaedigk",
    text:
      "Gaedigk, A. et al. (2018). The Pharmacogene Variation (PharmVar) Consortium: incorporation of the human CYP nomenclature database. Clin. Pharmacol. Ther. 103, 399–401.",
  },
];

// Map a reference key to its 1-based number in REFERENCES, so inline markers
// stay in sync with the list order.
const R = Object.fromEntries(REFERENCES.map((r, i) => [r.key, i + 1]));

export default function ProteinModelling() {
  return (
    <>
      <Head>
        <title>Modelling cytochrome P450 as a categorical knowledge-graph runtime · buhera</title>
        <meta
          name="description"
          content="A rigorous report: the P450 catalytic cycle expressed as a knowledge graph that is its own runtime, with a live in-browser IDE."
        />
      </Head>

      <div className="min-h-screen bg-black text-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-10">
          <nav className="mb-8 text-sm flex items-center justify-between">
            <Link href="/tutorials" className="text-blue-400 hover:text-blue-300">
              ← tutorials
            </Link>
            <Link href="/" className="text-blue-400 hover:text-blue-300">
              terminal →
            </Link>
          </nav>

          {/* ---- title block ---- */}
          <header className="mb-8">
            <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">
              Buhera OS · protein modelling report
            </p>
            <h1 className="text-4xl font-bold text-white leading-tight">
              A cytochrome&nbsp;P450 model in which the knowledge graph{" "}
              <span className="text-emerald-400">is</span> the runtime
            </h1>
            <p className="mt-4 text-gray-400 leading-relaxed">
              We express the catalytic cycle of cytochrome P450 as a categorical
              knowledge graph whose vertices are catalytic states and whose facts
              are the emitted output of a federation of domain solvers. The graph
              is not a description of a run — it is the run. Three domain-specific
              languages (an S-entropy redox solver, a mass-spectrometry acquisition
              DSL, and the native P450 module) fold their real output onto shared
              nodes of a single seven-state orbit. The apparatus is embedded live
              at the end of this report.
            </p>
          </header>

          {/* ---- abstract ---- */}
          <div className="border border-gray-800 rounded p-5 bg-gray-950">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-2">
              Abstract
            </h2>
            <p className="text-gray-300 text-sm leading-relaxed">
              A prior model discriminated reaction <em>participants</em> from{" "}
              <em>carriers</em> in transaminase catalysis using two kinds of fact,
              producing a graph of moderate vertex density. Cytochrome P450 offers
              a far richer corpus — a seven-state catalytic orbit, a three-cofactor
              electron-transfer chain, a reactive high-valent iron-oxo species,
              seven reaction families, three independent spectroscopic observables,
              and a pharmacogenomic manifold of 57 human isoforms. We port the
              participant/carrier discrimination and extend it: nine distinct fact
              families are folded onto one catalytic-cycle graph, several meeting on
              a single vertex, with two auxiliary DSLs (SBS, shapeshifter)
              contributing their whole solver output intact. The result is a denser
              graph carrying, on each vertex, the exact artifacts the interactive
              runtime renders. Every reported quantity is reproducible by running
              the corresponding file in the embedded IDE.
            </p>
          </div>

          {/* ---- 1. Introduction ---- */}
          <Section id="introduction" n="1" title="Introduction">
            <p>
              Cytochrome P450 enzymes are heme-thiolate monooxygenases responsible
              for the oxidative metabolism of the majority of clinically used
              drugs. Their catalytic cycle is unusually well-characterised: a
              resting ferric enzyme binds substrate, is reduced, binds dioxygen,
              and — after a second electron and two protons — forms{" "}
              <strong className="text-white">Compound I</strong>, an
              Fe(IV)=O porphyrin π-cation radical that abstracts a hydrogen atom
              from the substrate and rebounds to yield the hydroxylated product,
              returning the enzyme to rest. Sequence-level substrate-recognition
              and structural context for this cycle are well established
              <Cite n={R.gotoh} />
              <Cite n={R.berman} />
              <Cite n={R.varadi} />.
            </p>
            <p>
              Conventional models represent such a cycle as data: a diagram, a
              table of rate constants, a kinetic scheme integrated by a separate
              solver. The model and the computation are distinct artifacts. The
              premise of this work is different. We treat the catalytic cycle as a{" "}
              <strong className="text-white">
                categorical knowledge graph (CKG)
              </strong>{" "}
              whose reasoner <em>is</em> the runtime: a vertex is a catalytic
              state, a fact folded onto it is the value-delta emitted by a domain
              solver dispatched at that state, and the trajectory the runtime walks
              is the graph itself. There is no separate simulator to keep in sync,
              because the graph carries the computation.
            </p>
            <p>
              This report has one thesis to defend, inherited from a transaminase
              precedent and sharpened here:{" "}
              <strong className="text-white">
                a richer corpus should produce a richer graph
              </strong>
              . The transaminase model drew a single distinction from two fact
              kinds. If the framework is real, the P450 corpus — which carries far
              more than two kinds of fact — should yield a measurably denser graph:
              more fact families, more vertices carrying multiple facts, and
              contributions from more than one solver meeting on the same state.
            </p>
          </Section>

          {/* ---- 2. The corpus ---- */}
          <Section id="corpus" n="2" title="The corpus">
            <p>
              The model is grounded in a monograph of measured and computed P450
              quantities, in turn drawing on standard physicochemical descriptors
              <Cite n={R.kyte} />
              <Cite n={R.zamyatnin} /> and the human-isoform nomenclature
              <Cite n={R.gaedigk} />. We enumerate the fact families rather than
              sample them; each becomes an independently addressable fact in the
              runtime.
            </p>
            <div className="overflow-x-auto my-4">
              <table className="w-full text-sm text-left border border-gray-800">
                <thead className="bg-gray-900 text-gray-200">
                  <tr>
                    <th className="px-3 py-2 border-b border-gray-800">Fact family</th>
                    <th className="px-3 py-2 border-b border-gray-800">Key quantities</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  {[
                    ["Electron-transfer chain", "NADPH→FAD→FMN→heme; Marcus λ = 0.85 eV; coupling decay d_C = 4; FMN→heme rate-limiting (≈5×10⁶ s⁻¹)"],
                    ["Compound I", "Fe(IV)=O π-cation radical; aperture depth ΔM = ln2 ≈ 0.693; PCET/HAT; KIE ≈ 1.7"],
                    ["Catalytic orbit", "seven states, closed (state 7 → state 1); ΣΔM ≈ 4.963"],
                    ["Reaction families (×7)", "S-ox < N-ox < N-dealk < O-dealk < aliphatic < aromatic < desaturation, by aperture depth"],
                    ["Soret band", "417 nm (resting) → 392 nm (Compound I)"],
                    ["EPR", "g = (2.42, 2.25, 1.92) low-spin ferric"],
                    ["Resonance Raman", "Fe=O stretch 795 cm⁻¹ → 758 cm⁻¹ under ¹⁸O"],
                    ["Isoform manifold", "57 human CYPs; CYP2D6 {UM 0.27, EM 0.55, IM 0.75, PM 2.50}; CYP2C9*3 ΔM 3.60; CYP3A4 depth 5.69"],
                    ["Participant/carrier cut", "heme Fe is a carrier — present in every mechanism, absent from every stoichiometric equation"],
                    ["Conditioned floor β", "β = floor_disc + floor_Q + floor_conv; at depth 9 the conversion term dominates"],
                  ].map(([a, b], i) => (
                    <tr key={i} className="border-b border-gray-800">
                      <td className="px-3 py-2 align-top font-medium text-gray-200">{a}</td>
                      <td className="px-3 py-2 align-top">{b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* ---- 3. Methods ---- */}
          <Section id="methods" n="3" title="Methods">
            <h3 className="text-xl font-semibold text-gray-100 mt-2">
              3.1 The CKG runtime
            </h3>
            <p>
              An ontology maps each catalytic state to a node carrying a bag of{" "}
              <em>chunks</em>. Dispatching a node runs every chunk; each returns a{" "}
              value-delta <code className="text-emerald-300">{"{ kind, payload }"}</code>{" "}
              that is folded onto the node under a channel named by its{" "}
              <code className="text-emerald-300">kind</code>. The reasoner judges
              nothing — a derived fact is simply an emitted delta, and the graph
              that results is the trajectory of the run. A report projects the
              trajectory back out as a per-module dossier.
            </p>
            <p>
              A single change was required to admit a dense graph. Facts were
              previously keyed by module identity alone, so two chunks of one
              module on one node collided (last write wins). We re-key each fact by{" "}
              <code className="text-emerald-300">fact:&lt;module&gt;#&lt;chunk&gt;</code>,
              so a single vertex can carry many facts — from one module or several.
              This is what lets the Compound-I state hold its chemistry, its Fe=O
              Raman line, and a mass-spectrometry acquisition simultaneously.
            </p>

            <h3 className="text-xl font-semibold text-gray-100 mt-4">
              3.2 The federation and its DSLs
            </h3>
            <p>
              Three solvers contribute. The native{" "}
              <strong className="text-white">cytochrome</strong> module exposes
              every corpus fact family as an addressable operation. The{" "}
              <strong className="text-white">SBS</strong> DSL compiles an S-entropy
              circuit; we author the CPR→heme redox ladder as a four-node circuit
              and let its solver return coherence <em>R</em>, flux visibility{" "}
              <em>V</em>, and a backward navigation of the ladder. The{" "}
              <strong className="text-white">shapeshifter</strong> DSL compiles a
              mass-spectrometry acquisition stage; we author a positive-mode
              Orbitrap scan for a CYP substrate and its oxidised metabolite,
              producing a records-plus-partition-field workspace.
            </p>
            <p>
              Crucially, both auxiliary DSLs are given <em>P450-specific</em>{" "}
              scripts, not generic demos, and their whole output-delta is folded
              onto the graph intact — the SBS circuit and the mass-spec workspace
              survive on the vertex exactly as the terminal renders them.
            </p>

            <h3 className="text-xl font-semibold text-gray-100 mt-4">
              3.3 Constructing the trajectory
            </h3>
            <p>
              We represent the seven catalytic states as nodes and attach
              contributors: the closed orbit and EPR fingerprint at{" "}
              <em>resting</em>; the ET chain <em>and</em> its SBS redox circuit at{" "}
              <em>reduction</em>; the Soret shift at <em>oxygen-binding</em>; the
              Compound-I chemistry, the Fe=O Raman line, and the mass-spec
              acquisition at <em>compound-i</em>; a reaction aperture at{" "}
              <em>oxidation</em>; and the participant/carrier cut with a
              pharmacogenomic ΔM at <em>product-release</em>. Running the carry
              folds every contributor&rsquo;s output onto its state.
            </p>
          </Section>

          {/* ---- 4. Results ---- */}
          <Section id="results" n="4" title="Results">
            <p>
              The constructed graph holds{" "}
              <strong className="text-white">≥ 11 facts across seven vertices</strong>,
              spanning <strong className="text-white">≥ 9 distinct fact kinds</strong>,
              with contributions from{" "}
              <strong className="text-white">three modules</strong>. This is the
              transaminase precedent&rsquo;s two-fact graph made dense — the thesis of
              Section 1, made checkable. Three vertices carry multiple facts:
            </p>
            <div className="overflow-x-auto my-4">
              <table className="w-full text-sm text-left border border-gray-800">
                <thead className="bg-gray-900 text-gray-200">
                  <tr>
                    <th className="px-3 py-2 border-b border-gray-800">Catalytic state</th>
                    <th className="px-3 py-2 border-b border-gray-800">Facts folded</th>
                    <th className="px-3 py-2 border-b border-gray-800">Modules</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  {[
                    ["resting", "closed orbit (ΣΔM 4.963), EPR g-tensor", "cytochrome"],
                    ["reduction", "ET chain (λ 0.85 eV), SBS redox circuit (R, V)", "cytochrome + sbs"],
                    ["compound-i", "Fe=O chemistry (ΔM ln2), Raman 795→758 cm⁻¹, MS acquisition", "cytochrome + shapeshifter"],
                    ["product-release", "participant/carrier cut, CYP2D6 PM ΔM 2.50", "cytochrome"],
                  ].map(([a, b, c], i) => (
                    <tr key={i} className="border-b border-gray-800">
                      <td className="px-3 py-2 align-top font-mono text-gray-200">{a}</td>
                      <td className="px-3 py-2 align-top">{b}</td>
                      <td className="px-3 py-2 align-top text-gray-400">{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              The reaction families order correctly by aperture depth (S-oxidation
              &lt; N-oxidation &lt; N-dealkylation &lt; O-dealkylation &lt;
              aliphatic hydroxylation), and the kinetic-isotope diagnostic
              separates direct oxygen transfer (S-oxidation, KIE ≈ 1) from
              hydrogen-atom transfer (aliphatic hydroxylation, large KIE). The
              pharmacogenomic ΔM shifts order UM &lt; EM &lt; IM &lt; PM, and the
              CYP2C9*3 loss-of-function allele lands at ΔM 3.60. The
              participant/carrier invariant survives the port: the heme iron is
              reported as a carrier, present in every mechanism yet absent from
              every stoichiometric equation.
            </p>
            <p>
              The SBS contribution carries its full circuit and S-entropy metrics
              onto the <em>reduction</em> vertex — the same object the metrics
              dashboard renders — and the shapeshifter contribution carries its
              produced spectra workspace onto <em>compound-i</em>. The report
              projects all three modules&rsquo; whole deltas, not a tally. These are not
              claims about the runtime; they are the runtime&rsquo;s output, reproducible
              below.
            </p>
          </Section>

          {/* ---- 5. Discussion ---- */}
          <Section id="discussion" n="5" title="Discussion">
            <p>
              The density result is modest in absolute terms — eleven facts, nine
              kinds — but it is the <em>right</em> comparison: the same framework,
              the same fold mechanism, a richer corpus, and a demonstrably denser
              graph, with the one structural change (per-chunk fact keying) that a
              denser graph logically requires. Nothing here is authored into a
              schema; the vertices are catalytic states, contributors are attached,
              the carry runs, and the trajectory is read back.
            </p>
            <p>
              That two independent DSLs fold P450-specific output onto the same
              graph is the load-bearing point. It shows the corpus is expressible
              across the whole federation rather than trapped in one bespoke
              module: a redox circuit and a mass-spectrometry acquisition, written
              in their own languages, become facts on the same catalytic cycle as
              the native chemistry. The graph is the meeting ground.
            </p>
            <p>
              The runtime stores no fact dictionary — a derived fact is recovered
              by running its chunk, never looked up. This is not a slogan: two
              on-disk grounding papers make it measurable. An empty-dictionary
              scheme answers 6 of 6 P450 queries against 0 of 6 for an index
              control while its resident state stays a constant 561 bytes across
              nine orders of corpus size, and its graded response correlates with
              catalytic depth (ρ ≈ 0.382 over 190 pairs, against a shuffle-null
              mean near zero) <Cite n={R["empty-dict"]} />; the companion
              database-recovery paper reconstructs P450 records from partial
              ternary addresses by constraint propagation, with nothing stored
              <Cite n={R["db-recovery"]} />. This is the same discipline the
              graph runs on: every vertex in the notebook below holds its
              contributors as chunks, not as cached values, and each fact appears
              only when its cell runs the chunk that emits it. The depth
              correlation is a rank statistic <Cite n={R.spearman} />, reported
              honestly with its null and its non-discriminating identity control.
            </p>
            <p>
              <strong className="text-white">Limitations.</strong> The SBS solver
              falls back to a CPU integrator when its accelerated backend is
              absent; the metrics are real but backend-dependent. The scope module
              is not exercised here (its compiler does not load outside the
              browser bundle). The corpus is a monograph snapshot, not a primary
              re-derivation.
            </p>
          </Section>

          {/* ---- 6. The apparatus ---- */}
          <Section id="apparatus" n="6" title="The apparatus">
            <p>
              Everything above is reproducible in the notebook below. It runs like
              a Jupyter notebook: a setup cell brings the federation up and clears
              the runtime, then each cell holds one real script and renders that
              script&rsquo;s own output at its foot. Run the cells top to bottom —
              they share a single kernel, so state flows downstream exactly as in a
              notebook. Cells{" "}
              <span className="text-emerald-300 font-mono">In[1]</span>–
              <span className="text-emerald-300 font-mono">In[3]</span> run the
              three DSLs on their own (the SBS redox circuit, the shapeshifter
              mass-spec acquisition, and each cytochrome corpus fact family). Cell{" "}
              <span className="text-emerald-300 font-mono">In[4]</span> folds all
              three onto one seven-state graph; cell{" "}
              <span className="text-emerald-300 font-mono">In[5]</span> reads the
              trajectory back — the same graph In[4] built, not a fresh one. Every
              number in Section&nbsp;4 is one a cell here reproduces.
            </p>
          </Section>

          <div className="mt-6">
            <P450Notebook />
          </div>

          <p className="mt-4 text-xs text-gray-600">
            The notebook runs entirely in your browser against the in-page
            federation. Because the cells share one kernel, the read-back cells
            (In[5]) assume you have run the build cell (In[4]) above them first in
            this session — just as a notebook cell depends on the cells above it.
          </p>

          {/* ---- References ---- */}
          <Section id="references" n="7" title="References">
            <ol className="list-decimal list-outside ml-5 space-y-2 text-sm text-gray-400 marker:text-gray-600">
              {REFERENCES.map((r) => (
                <li key={r.key} className="pl-1 leading-relaxed">
                  {r.text}
                </li>
              ))}
            </ol>
            <p className="mt-4 text-xs text-gray-600 leading-relaxed">
              References [1] and [2] are the on-disk grounding papers: every
              quantity attributed to them is reproduced by their validation
              scripts, which the notebook&rsquo;s cytochrome and CKG cells re-run
              against the live federation. Two honesty flags carried from paper [1]
              rather than smoothed over: its body labels it &ldquo;Paper 16&rdquo;
              while its conclusion calls itself the seventeenth paper, and its
              graded-response p-value is reported as p&nbsp;&lt;&nbsp;0.005 where the
              underlying results file records p&nbsp;=&nbsp;0.0.
            </p>
          </Section>

          <hr className="my-10 border-gray-800" />
          <nav className="flex items-center justify-between text-sm">
            <Link href="/tutorials" className="text-blue-400 hover:text-blue-300">
              ← all tutorials
            </Link>
            <Link href="/" className="text-blue-400 hover:text-blue-300">
              open the terminal →
            </Link>
          </nav>
        </div>
      </div>
    </>
  );
}
