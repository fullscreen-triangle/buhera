/* ============================================================================
 * The P450 IDE file catalog.
 *
 * Every "file" is a real script whose `source` is fed verbatim to runInput()
 * — the identical entry point the terminal and every tutorial RunnableCell
 * use. So "running a file" in the IDE dispatches against the live federation
 * and renders the DSL's own artifact, exactly as it renders in the terminal.
 *
 * Folders mirror the three DSLs the P450 model is expressed through, plus a
 * `cytochrome/` folder for the native module and a `ckg/` folder that folds
 * all three onto one catalytic-cycle graph (the dense-trajectory result).
 *
 * `source` is a terminal command string. The universal driver is
 *   dispatch("<module>", <instruction>)
 * which routes to any federation module and returns its output_delta as an
 * Artifact — so a "file" is just that call with the DSL script inlined.
 * ========================================================================== */

import { P450_ET_SBS, P450_MS_SS } from "@/lib/modules/cytochrome-module";

// A dispatch call with a multi-line script argument. We JSON-encode the script
// so newlines/quotes survive the `dispatch("mod", "...")` parser.
function dispatchCall(moduleId, script) {
  return `dispatch("${moduleId}", ${JSON.stringify(script)})`;
}

// A dispatch call whose instruction is a structured object (op-style modules).
function dispatchOp(moduleId, obj) {
  return `dispatch("${moduleId}", ${JSON.stringify(obj)})`;
}

export const IDE_FILES = [
  {
    folder: "sbs",
    icon: "⚡",
    hint: "S-entropy biological solver — redox circuits",
    files: [
      {
        name: "electron-transfer.sbs",
        lang: "sbs",
        blurb:
          "The CPR→heme redox ladder (NADPH→FAD→FMN→heme) as an S-entropy circuit. " +
          "Running it solves for coherence R and flux visibility V, then navigates the " +
          "ladder backward from the heme.",
        // The raw script, shown in the editor…
        editor: P450_ET_SBS,
        // …and the command that actually runs it.
        source: dispatchCall("sbs", P450_ET_SBS),
      },
    ],
  },
  {
    folder: "shapeshifter",
    icon: "🔬",
    hint: "Lavoisier mass-spectrometry acquisition DSL",
    files: [
      {
        name: "metabolite-scan.ss",
        lang: "shapeshifter",
        blurb:
          "A positive-mode Orbitrap acquisition for a CYP substrate and its oxidised " +
          "metabolite. Compiling and running the stage produces a records + partition-field " +
          "workspace — the same spectra the model folds onto the Compound-I node.",
        editor: P450_MS_SS,
        source: dispatchCall("shapeshifter", P450_MS_SS),
      },
    ],
  },
  {
    folder: "cytochrome",
    icon: "🧬",
    hint: "The native P450 module — the monograph, addressable",
    files: [
      {
        name: "electron-transfer.cyp",
        lang: "cytochrome",
        blurb:
          "The three-hop ET chain: Marcus λ, the coupling decay d_C, and the " +
          "rate-limiting FMN→heme step.",
        editor: `cytochrome electron-transfer\n// → the NADPH→FAD→FMN→heme chain, λ = 0.85 eV, d_C = 4`,
        source: dispatchOp("cytochrome", { op: "electron-transfer" }),
      },
      {
        name: "compound-i.cyp",
        lang: "cytochrome",
        blurb:
          "The reactive Fe(IV)=O porphyrin cation radical: a depth-ln2 aperture, PCET " +
          "hydrogen-atom transfer, KIE ≈ 1.7.",
        editor: `cytochrome compound-i\n// → ΔM = ln2, PCET/HAT, KIE ≈ 1.7`,
        source: dispatchOp("cytochrome", { op: "compound-i" }),
      },
      {
        name: "catalytic-orbit.cyp",
        lang: "cytochrome",
        blurb:
          "The seven-state catalytic cycle as a closed orbit on the state manifold; " +
          "the aperture depths sum to ΣΔM ≈ 4.963 and the orbit returns to resting.",
        editor: `cytochrome states\n// → seven states, closed orbit, ΣΔM ≈ 4.963`,
        source: dispatchOp("cytochrome", { op: "states" }),
      },
      {
        name: "reaction-hydroxylation.cyp",
        lang: "cytochrome",
        blurb:
          "One of seven reaction families. Aliphatic hydroxylation is a hydrogen-atom " +
          "abstraction — hence the large kinetic isotope effect.",
        editor: `cytochrome pathway aliphatic-hydroxylation\n// → HAT aperture, large KIE`,
        source: dispatchOp("cytochrome", {
          op: "pathway",
          reaction: "aliphatic-hydroxylation",
        }),
      },
      {
        name: "spectroscopy-soret.cyp",
        lang: "cytochrome",
        blurb:
          "The Soret band: 417 nm at rest, shifting to 392 nm at Compound I — an " +
          "optical readout of the aperture opening.",
        editor: `cytochrome soret\n// → 417 nm (resting) → 392 nm (Compound I)`,
        source: dispatchOp("cytochrome", { op: "soret" }),
      },
      {
        name: "pharmacogenomics-2d6.cyp",
        lang: "cytochrome",
        blurb:
          "CYP2D6 phenotype classes as ΔM shifts on the allele: ultrarapid, extensive, " +
          "intermediate, poor. A poor metaboliser is the largest shift.",
        editor: `cytochrome isoform CYP2D6 PM\n// → poor-metaboliser ΔM = 2.50`,
        source: dispatchOp("cytochrome", {
          op: "isoform",
          cyp: "CYP2D6",
          phenotype: "PM",
        }),
      },
      {
        name: "participants.cyp",
        lang: "cytochrome",
        blurb:
          "The participant/carrier discrimination — the transaminase precedent, ported. " +
          "The heme iron never appears in a stoichiometric equation, so it is a carrier, " +
          "not a participant.",
        editor: `cytochrome participants aliphatic-hydroxylation\n// → heme Fe: carrier, not participant`,
        source: dispatchOp("cytochrome", {
          op: "participants",
          reaction: "aliphatic-hydroxylation",
        }),
      },
    ],
  },
  {
    folder: "ckg",
    icon: "🕸",
    hint: "The knowledge graph that IS the runtime",
    files: [
      {
        name: "build-cycle.ckg",
        lang: "ckg",
        blurb:
          "Represent the seven catalytic stages as nodes, then attach cytochrome, sbs, " +
          "and shapeshifter contributors. Several facts land on the same stage — the ET " +
          "chain and its SBS circuit both on `reduction`, the Compound-I chemistry, its " +
          "Fe=O Raman line, and the mass-spec acquisition all on `compound-i`.",
        // A multi-command script — runInput runs one line; the IDE runs each
        // line in sequence and shows the last artifact. See P450Ide.runFile.
        editor: [
          'ckg reset',
          'ckg represent resting',
          'ckg represent reduction',
          'ckg represent compound-i',
          'ckg attach reduction et cytochrome {op:"electron-transfer"}',
          'ckg attach reduction redox sbs <electron-transfer.sbs>',
          'ckg attach compound-i cpdI cytochrome {op:"compound-i"}',
          'ckg attach compound-i ms shapeshifter <metabolite-scan.ss>',
          'ckg carry',
        ].join("\n"),
        // The executable form: a sequence of dispatches to the ckg module.
        program: [
          dispatchOp("ckg", { op: "reset" }),
          ...[
            "resting", "substrate-bound", "reduction", "oxygen-binding",
            "compound-i", "oxidation", "product-release",
          ].map((tau) =>
            dispatchOp("ckg", { op: "represent", tau, seed: tau === "resting" ? 2 : 0 })
          ),
          dispatchOp("ckg", { op: "attach", tau: "resting", name: "orbit", module: "cytochrome", instruction: { op: "states" } }),
          dispatchOp("ckg", { op: "attach", tau: "resting", name: "epr", module: "cytochrome", instruction: { op: "epr" } }),
          dispatchOp("ckg", { op: "attach", tau: "reduction", name: "et", module: "cytochrome", instruction: { op: "electron-transfer" } }),
          dispatchOp("ckg", { op: "attach", tau: "reduction", name: "redox", module: "sbs", instruction: P450_ET_SBS }),
          dispatchOp("ckg", { op: "attach", tau: "oxygen-binding", name: "soret", module: "cytochrome", instruction: { op: "soret" } }),
          dispatchOp("ckg", { op: "attach", tau: "compound-i", name: "cpdI", module: "cytochrome", instruction: { op: "compound-i" } }),
          dispatchOp("ckg", { op: "attach", tau: "compound-i", name: "raman", module: "cytochrome", instruction: { op: "raman" } }),
          dispatchOp("ckg", { op: "attach", tau: "compound-i", name: "ms", module: "shapeshifter", instruction: P450_MS_SS }),
          dispatchOp("ckg", { op: "attach", tau: "oxidation", name: "pathway", module: "cytochrome", instruction: { op: "pathway", reaction: "aliphatic-hydroxylation" } }),
          dispatchOp("ckg", { op: "attach", tau: "product-release", name: "parts", module: "cytochrome", instruction: { op: "participants", reaction: "aliphatic-hydroxylation" } }),
          dispatchOp("ckg", { op: "attach", tau: "product-release", name: "pgx", module: "cytochrome", instruction: { op: "isoform", cyp: "CYP2D6", phenotype: "PM" } }),
          dispatchOp("ckg", { op: "carry" }),
        ],
        // After building, show the graph.
        source: dispatchOp("ckg", { op: "graph" }),
      },
      {
        name: "graph.ckg",
        lang: "ckg",
        blurb:
          "Read back the whole trajectory: the seven nodes, the facts folded on each, and " +
          "the emergent edges. Run build-cycle.ckg first so there is a graph to read.",
        editor: `ckg graph\n// → nodes, folded facts, emergent edges`,
        source: dispatchOp("ckg", { op: "graph" }),
      },
      {
        name: "report.ckg",
        lang: "ckg",
        blurb:
          "The dossier: every contributing module, its whole output_delta carried intact " +
          "(the SBS circuit, the mass-spec workspace), and an extracted findings headline " +
          "per fact. Run build-cycle.ckg first.",
        editor: `ckg report\n// → per-module dossier with the full deltas`,
        source: dispatchOp("ckg", { op: "report" }),
      },
    ],
  },
];

// Flatten to a { path -> file } lookup for the editor.
export function fileByPath(path) {
  for (const group of IDE_FILES) {
    for (const f of group.files) {
      if (`${group.folder}/${f.name}` === path) return f;
    }
  }
  return null;
}

export const DEFAULT_FILE = "ckg/build-cycle.ckg";
