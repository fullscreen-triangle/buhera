/* ============================================================================
 * P450Notebook — the P450 model laid out as a Jupyter-style notebook.
 *
 * Where P450Ide is a file-picker (run one script at a time), this is a linear
 * notebook: a setup/imports cell at the top, then one cell per script stacked
 * top-to-bottom, each rendering its own output at its bottom. Every cell runs
 * against a SINGLE shared runtime context (`ctxRef`), so state flows downstream
 * exactly as in a real notebook kernel — the graph the build-cycle cell folds
 * is the same graph the graph/report cells below read back.
 *
 * Each cell is a real script drawn from the same IDE_FILES catalog the IDE
 * uses; the source is fed verbatim to runInput() (via RunnableCell), the
 * identical entry point the terminal and every tutorial cell use. So the output
 * rendered under each cell is byte-for-byte the federation's own artifact.
 * ========================================================================== */

import { useEffect, useRef, useState } from "react";
import RunnableCell from "@/components/RunnableCell";
import { createRuntimeContext } from "@/lib/runtime/run-input";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { IDE_FILES, fileByPath } from "@/lib/p450-ide-files";

// A cell's runnable source. For build-cycle.ckg the executable form is its
// `program` (a sequence of dispatches) followed by the shown `source`; joining
// them with newlines lets runInput() run the whole thing as one multi-statement
// cell — the same rescue path the terminal takes for a pasted script.
function cellSource(file) {
  const program = file.program || [];
  return [...program, file.source].join("\n");
}

// The ordered notebook: the file catalog flattened into (heading, cells) blocks
// in the order the report walks them — the three DSLs on their own, then the
// ckg cells that fold them onto one graph.
const NB_BLOCKS = [
  {
    n: "In[1]",
    heading: "1 · The redox circuit (SBS)",
    intro:
      "The CPR→heme electron-transfer ladder as an S-entropy circuit. Running it " +
      "solves for coherence R and flux visibility V — the same delta that later " +
      "folds onto the reduction node.",
    paths: ["sbs/electron-transfer.sbs"],
  },
  {
    n: "In[2]",
    heading: "2 · The mass-spectrometry acquisition (shapeshifter)",
    intro:
      "A positive-mode Orbitrap scan for a CYP substrate and its oxidised " +
      "metabolite. The compiled stage produces a records + partition-field " +
      "workspace — the spectra that fold onto the Compound-I node.",
    paths: ["shapeshifter/metabolite-scan.ss"],
  },
  {
    n: "In[3]",
    heading: "3 · The native corpus (cytochrome)",
    intro:
      "Each cell addresses one fact family of the monograph directly: the ET " +
      "chain, Compound I, the seven-state orbit, a reaction aperture, the Soret " +
      "band, a pharmacogenomic ΔM, and the participant/carrier cut.",
    paths: [
      "cytochrome/electron-transfer.cyp",
      "cytochrome/compound-i.cyp",
      "cytochrome/catalytic-orbit.cyp",
      "cytochrome/reaction-hydroxylation.cyp",
      "cytochrome/spectroscopy-soret.cyp",
      "cytochrome/pharmacogenomics-2d6.cyp",
      "cytochrome/participants.cyp",
    ],
  },
  {
    n: "In[4]",
    heading: "4 · Fold them onto one graph (CKG)",
    intro:
      "Represent the seven catalytic stages as nodes, attach the cytochrome, " +
      "sbs, and shapeshifter contributors, and run the carry. Several facts land " +
      "on one stage — the ET chain and its SBS circuit both on reduction; the " +
      "Compound-I chemistry, its Fe=O Raman line, and the mass-spec acquisition " +
      "all on compound-i. This is the downstream-state cell: everything below " +
      "reads the graph it builds.",
    paths: ["ckg/build-cycle.ckg"],
  },
  {
    n: "In[5]",
    heading: "5 · Read the trajectory back (CKG)",
    intro:
      "The graph and the report project the run back out — the seven nodes with " +
      "their folded facts and emergent edges, then a per-module dossier carrying " +
      "each contributor's whole delta intact. Run cell 4 first so there is a " +
      "graph to read.",
    paths: ["ckg/graph.ckg", "ckg/report.ckg"],
  },
];

function Cell({ file, ctxRef, ready }) {
  return (
    <div className="mb-6">
      {file.blurb && (
        <p className="text-xs text-gray-500 leading-relaxed mb-1 font-mono">
          # {file.name}
        </p>
      )}
      {ready ? (
        <RunnableCell source={cellSource(file)} ctxRef={ctxRef} />
      ) : (
        <div className="my-4 border border-gray-800 rounded bg-gray-900 p-3 text-xs text-gray-600 font-mono">
          booting kernel…
        </div>
      )}
    </div>
  );
}

export default function P450Notebook() {
  const ctxRef = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    ctxRef.current = createRuntimeContext();
    bootstrapFederation();
    setReady(true);
  }, []);

  return (
    <div className="rounded-lg overflow-hidden border border-[#2b2b2b] bg-[#151515]">
      {/* notebook toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 bg-[#1f1f1f] border-b border-[#2b2b2b] text-xs">
        <span className="text-gray-300 font-mono">
          p450-model.ipynb
        </span>
        <span className="ml-auto text-gray-500 font-mono">
          {ready ? "● kernel: buhera-federation (idle)" : "starting kernel…"}
        </span>
      </div>

      <div className="p-4 md:p-6 space-y-2">
        {/* ---- setup / imports cell ---- */}
        <div className="mb-6">
          <p className="text-xs text-gray-500 leading-relaxed mb-1 font-mono">
            # setup — boot the federation and clear any prior graph
          </p>
          <p className="text-[13px] text-gray-400 leading-relaxed mb-2">
            Like the imports at the top of a notebook, this cell brings the
            federation up and resets the CKG runtime, so the cells below start
            from a clean kernel. Run it first.
          </p>
          {ready ? (
            <RunnableCell
              source={[
                ':modules',
                'dispatch("ckg", { op: "reset" })',
              ].join("\n")}
              ctxRef={ctxRef}
            />
          ) : (
            <div className="my-4 border border-gray-800 rounded bg-gray-900 p-3 text-xs text-gray-600 font-mono">
              booting kernel…
            </div>
          )}
        </div>

        {/* ---- one block per DSL / stage ---- */}
        {NB_BLOCKS.map((block) => (
          <section key={block.n} className="pt-2">
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-[11px] font-mono text-gray-600">{block.n}</span>
              <h3 className="text-lg font-semibold text-gray-100">
                {block.heading}
              </h3>
            </div>
            <p className="text-[13px] text-gray-400 leading-relaxed mb-3">
              {block.intro}
            </p>
            {block.paths.map((path) => {
              const file = fileByPath(path);
              if (!file) return null;
              return (
                <div key={path}>
                  {file.blurb && (
                    <p className="text-xs text-gray-500 leading-relaxed mb-2 pl-1">
                      {file.blurb}
                    </p>
                  )}
                  <Cell file={file} ctxRef={ctxRef} ready={ready} />
                </div>
              );
            })}
          </section>
        ))}
      </div>
    </div>
  );
}
