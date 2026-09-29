import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { useGatewaySession } from "@/lib/auth/useGatewaySession";
import { Kernel } from "@/lib/kernel";
import { embedProtein } from "@/lib/substrate";
import { translate } from "@/lib/translator";
import { executeVahera } from "@/lib/vahera";
import { PROTEINS } from "@/lib/proteins";
import { run as runTurbulance } from "@/lib/turbulance";
import { listModules, dispatch as dispatchModule, getAuditLog } from "@/lib/modules/registry";
import { linkScopeImage } from "@/lib/modules/scope-module";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { routeInput } from "@/lib/runtime/route-input";
import { Artifact } from "@/components/artifacts/Artifact";

// The renderers moved to components/artifacts/Artifact.js; re-exported so
// existing importers of { Artifact } from this file keep working.
export { Artifact };

// ────────────────────────────────────────────────────────────
//  Kernel boot.
// ────────────────────────────────────────────────────────────

function bootBlank() {
  return new Kernel(12);
}

function loadProteins(kernel) {
  for (const name of Object.keys(PROTEINS)) {
    const coord = embedProtein(name, PROTEINS[name]);
    kernel.allocate(coord, PROTEINS[name], {
      name,
      gene: PROTEINS[name].gene,
      kind: "protein",
    });
  }
}

// ────────────────────────────────────────────────────────────
//  Input router.
//
//  Returns { type, vahera?, meta? }
//    type === "vahera" — `vahera` is source to execute
//    type === "meta"   — `meta` is one of "tour" | "proteins" | "clear"
//                                       | "help"
//    type === "nl"     — caller should fall back to the NL translator
// ────────────────────────────────────────────────────────────

/**
 * Decode a PNG/JPEG image URL to a grayscale ImagePayload for the SCOPE runtime.
 * Reuses the browser-native decode chain (fetch → blob → createImageBitmap →
 * canvas → getImageData), then collapses RGBA to a single luminance channel.
 * TIFF is not supported — browsers cannot decode it via createImageBitmap.
 */
async function loadImagePayload(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();

  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    throw new Error("unsupported image format (use PNG or JPEG)");
  }

  const { width, height } = bitmap;
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement("canvas"), { width, height });
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d canvas context");
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, width, height);

  // RGBA → grayscale Float32Array (Rec. 601 luma), normalised to [0,1].
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
  }
  return { data: gray, width, height };
}

// The line router and dispatch-call parser live in a pure module so the
// Node-side runner and tests can share them. Re-exported here so existing
// importers of routeInput/parseDispatchCall from this file keep working.
export { routeInput, parseDispatchCall } from "@/lib/runtime/route-input";

const TOUR_VAHERA = `
memory store "weekend"   = "I need to do laundry and clean the kitchen this weekend"
memory store "groceries" = "buy milk eggs bread and coffee from the supermarket"
memory store "exercise"  = "go for a run on Saturday morning before it gets hot"
memory store "travel"    = "book a flight to Munich for the conference next month"
memory store "code"      = "refactor the database connection pool to use async"
memory find nearest "shopping list" k=3
memory find nearest "morning workout" k=3
memory find nearest "flight to Germany" k=3
kernel stats
`.trim();

// ────────────────────────────────────────────────────────────
//  Welcome banner.
// ────────────────────────────────────────────────────────────

const WELCOME = `\
buhera-os web demo · in-browser kernel, no install

what it does
  files text by its categorical address — three numbers
  derived from the meaning of what you write. ask later,
  it finds what you stored, ranked by closeness.

quick try
  store note  = "remember to write up the proposal"
  store other = "buy milk and bread from the corner shop"
  find "writing"
  find "shopping"

other commands
  list                    show everything you've stored
  stats                   per-subsystem statistics
  dump <name>             show one object in detail
  sort                    zero-cost categorical sort
  :modules                list the federation
  :audit                  show recent acts
  :tutorials              open the tutorial index
  :experiment             open the CKG experiment report + notebook
  :tour                   load five sample notes and search them
  :proteins               load a hardcoded biology database for the
                          natural-language demo
  :clear                  reset the kernel
  :help                   show every vaHera statement

you can also type vaHera directly:
  memory store "name" = "text"
  memory find nearest "query" k=5
  describe X with "text", spawn p from X, navigate to penultimate,
  complete trajectory, kernel stats, controller verify ...

or a turbulance (kwasa-kwasa) script:
  funxn double(x): return x * 2
  item r = double(21)
  proposition Greeting: motion Hello("world")

  // call any registered Buhera module:
  item out = dispatch("echo", "hello federation")
  print(out.output_delta.value)

  // route vaHera through the orchestrator:
  item r = dispatch("vahera", "memory store \"n\" = \"hi\"")

  // run a virtual mass-spec experiment (lavoisier):
  item ms = dispatch("lavoisier", "demo")
  print("records: {}", ms.output_delta.summary.count)

  // run an individuation-theoretic search (graffiti / .grf):
  item g = dispatch("graffiti", "demo")
  print("floor: {}", g.output_delta.ambient_floor)

  // ask a natural-language question via the MSI (zangalewa; needs an LLM key —
  // OLLAMA_URL, GEMINI_API_KEY, or OPENAI_API_KEY):
  item z = dispatch("zangalewa", "what is p53?")
  print(z.output_delta.title)

  // compute the minimum-sufficient carry toward a goal (purpose):
  item c = dispatch("purpose-carry", { kind: "carry", goal: ["TUM", "AIMe"] })
  print("keep: {}", c.output_delta.keep)
`;

const HELP = `\
vaHera statements (15):
  describe <name> with "<text>"
  resolve <name>
  spawn <program> from <name>
  navigate to penultimate
  complete trajectory
  memory create at S(<k>,<t>,<e>)
  memory store "<name>" = "<text>"
  memory find nearest "<text>" k=<n>
  memory list
  memory dump <name>
  demon sort
  controller verify
  kernel stats
  kernel trace
  process list

shortcuts:
  store <name> = "<text>"
  find "<text>" [k=N]
  list / dump <name> / sort / stats / trace / procs / verify

meta:
  :tour  :proteins  :clear  :help  :quit
`;


// ────────────────────────────────────────────────────────────
//  Welcome panel.
// ────────────────────────────────────────────────────────────

function WelcomePanel() {
  return (
    <div>
      {/* The persistent top-right "▶ tutorials" link (rendered by the terminal
          shell) is the single entry point — the welcome panel does not repeat
          it. */}
      <div className="mb-6">
        <div className="text-white text-lg font-mono">buhera OS</div>
        <div className="text-gray-500 text-xs mt-0.5">a research operating system</div>
      </div>
      <pre className="text-gray-400 text-xs leading-relaxed whitespace-pre-wrap font-mono">
{WELCOME}
      </pre>
    </div>
  );
}

// ────────────────────────────────────────────────────────────
//  Terminal.
// ────────────────────────────────────────────────────────────

export default function BuheraTerminal() {
  const router = useRouter();
  const { email, logout } = useGatewaySession();
  const kernelRef = useRef(null);
  const inputRef = useRef(null);
  const historyRef = useRef(null);
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [proteinsMode, setProteinsMode] = useState(false);
  const [history, setHistory] = useState([]);
  const [cursor, setCursor] = useState(-1);

  useEffect(() => {
    kernelRef.current = bootBlank();
    // Register the whole federation + hooks in one call. Idempotent — the
    // tutorial pages call the same function.
    const cleanup = bootstrapFederation();
    return () => { cleanup(); };
  }, []);

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
    const onClick = () => inputRef.current && inputRef.current.focus();
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [entries]);

  function pushEntry(entry) {
    setEntries((e) => [...e, { id: Date.now() + Math.random(), ...entry }]);
  }

  async function dispatch(text) {
    const route = routeInput(text);
    if (route.type === "noop") return;

    pushEntry({ obs: text, thinking: true });

    // Small artificial delay so the screen doesn't flicker.
    await new Promise((r) => setTimeout(r, 80));

    function patchLast(patch) {
      setEntries((es) => {
        const out = [...es];
        for (let i = out.length - 1; i >= 0; i--) {
          if (out[i].obs === text && out[i].thinking) {
            out[i] = { ...out[i], thinking: false, ...patch };
            break;
          }
        }
        return out;
      });
    }

    try {
      if (route.type === "meta") {
        if (route.meta === "help") {
          patchLast({ result: { kind: "text", lines: HELP.split("\n") } });
        } else if (route.meta === "clear") {
          kernelRef.current = bootBlank();
          setProteinsMode(false);
          patchLast({ result: { kind: "text", lines: ["(kernel reset)"] } });
        } else if (route.meta === "tour") {
          const out = executeVahera(TOUR_VAHERA, kernelRef.current, {
            useProteinDb: false,
            rerank: true,
          });
          patchLast({ multi: out.results });
        } else if (route.meta === "proteins") {
          if (!proteinsMode) {
            loadProteins(kernelRef.current);
            setProteinsMode(true);
          }
          patchLast({
            result: {
              kind: "text",
              lines: [
                "(proteins demo loaded; try \"tell me about TP53\" or \"compare BRCA1 and BRCA2\")",
              ],
            },
          });
        } else if (route.meta === "modules") {
          const mods = listModules();
          const lines = mods.length === 0
            ? ["(no modules registered)"]
            : mods.flatMap((m) => [
                `[${m.id}]${m.description ? "  " + m.description : ""}`,
                ...(m.instructions || []).map((i) => "    " + i),
              ]);
          patchLast({ result: { kind: "text", lines } });
        } else if (route.meta === "audit") {
          const log = getAuditLog().slice(-15);
          const lines = log.length === 0
            ? ["(audit log is empty)"]
            : log.map((e) => `#${e.act_id} ${e.module_id} (${e.wall_clock_ms}ms) — ${
                typeof e.instruction === "string"
                  ? e.instruction.slice(0, 60)
                  : "[non-string instruction]"
              }`);
          patchLast({ result: { kind: "text", lines } });
        } else if (route.meta === "tutorials") {
          // Open the tutorials index in a new tab so the terminal session
          // is preserved. Fall back to same-tab navigation if popups blocked.
          if (typeof window !== "undefined") {
            const w = window.open("/tutorials", "_blank", "noopener");
            if (!w) {
              window.location.href = "/tutorials";
            }
          }
          patchLast({
            result: {
              kind: "text",
              lines: [
                "opening tutorials …",
                "if a new tab did not open, go to /tutorials directly.",
              ],
            },
          });
        } else if (route.meta === "experiment") {
          // Open the CKG experiment report + notebook in a new tab so the
          // terminal session is preserved. Fall back to same-tab navigation.
          if (typeof window !== "undefined") {
            const w = window.open("/protein-modelling", "_blank", "noopener");
            if (!w) {
              window.location.href = "/protein-modelling";
            }
          }
          patchLast({
            result: {
              kind: "text",
              lines: [
                "opening the CKG experiment …",
                "if a new tab did not open, go to /protein-modelling directly.",
              ],
            },
          });
        } else if (route.meta === "quit") {
          patchLast({ result: { kind: "text", lines: ["(can't quit a browser tab from here)"] } });
        }
        return;
      }

      if (route.type === "vahera") {
        const out = executeVahera(route.vahera, kernelRef.current, {
          useProteinDb: proteinsMode,
          rerank: true,
        });
        if (out.results.length === 1) {
          patchLast({ result: out.results[0] });
        } else if (out.results.length > 1) {
          patchLast({ multi: out.results });
        } else if (out.lastResult) {
          patchLast({ result: out.lastResult });
        } else {
          patchLast({ result: { kind: "text", lines: ["ok"] } });
        }
        return;
      }

      if (route.type === "turbulance") {
        const tb = await runTurbulance(route.source);
        patchLast({ result: { kind: "turbulance_result", tb } });
        return;
      }

      if (route.type === "scope_ctl") {
        if (route.ctl === "load") {
          if (!route.url) {
            patchLast({ result: { kind: "text", lines: ["usage: :scope load <png-or-jpeg-url>"] } });
            return;
          }
          try {
            const payload = await loadImagePayload(route.url);
            linkScopeImage(payload);
            patchLast({
              result: {
                kind: "text",
                lines: [`scope: image linked (${payload.width}×${payload.height}) from ${route.url}`],
              },
            });
          } catch (err) {
            patchLast({ result: { kind: "text", lines: [`scope: could not load image — ${err.message || String(err)}`] } });
          }
          return;
        }
        // reset / state go through the module so the audit log sees them.
        const instr = route.ctl === "reset" ? { kind: "reset" } : { kind: "state" };
        const res = await dispatchModule("scope", instr);
        patchLast({ result: res.output_delta });
        return;
      }

      if (route.type === "scope") {
        const res = await dispatchModule("scope", route.source);
        patchLast({ result: res.output_delta });
        return;
      }

      if (route.type === "srn") {
        const res = await dispatchModule("srn", route.instruction);
        patchLast({ result: res.output_delta });
        return;
      }

      if (route.type === "dispatch") {
        const res = await dispatchModule(route.moduleId, route.instruction);
        if (!res || res.output_delta == null) {
          patchLast({ result: { kind: "text", lines: [`(${route.moduleId}: no output)`] } });
        } else {
          patchLast({ result: res.output_delta });
        }
        return;
      }

      // NL input.
      if (proteinsMode) {
        // Use the legacy translator for the proteins demo.
        const vh = translate(route.text);
        const out = executeVahera(vh, kernelRef.current, {
          useProteinDb: true,
          rerank: true,
        });
        if (out.lastResult) {
          patchLast({ result: out.lastResult });
        } else if (out.results.length) {
          patchLast({ multi: out.results });
        } else {
          patchLast({ result: { kind: "text", lines: ["no categorical match."] } });
        }
        return;
      }

      // Otherwise: bare line, no proteins mode → search.
      const safe = route.text.replace(/"/g, "'");
      const vh = `memory find nearest "${safe}" k=3`;
      const out = executeVahera(vh, kernelRef.current, {
        useProteinDb: false,
        rerank: true,
      });
      patchLast({ result: out.lastResult });
    } catch (err) {
      patchLast({ error: err.message || String(err) });
    }
  }

  async function submit(text) {
    if (!text.trim() || busy) return;
    setBusy(true);
    setDraft("");
    setHistory((h) => (h[h.length - 1] === text ? h : [...h, text]));
    setCursor(-1);

    await dispatch(text);

    setBusy(false);
    if (inputRef.current) inputRef.current.focus();
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(draft);
    } else if (e.key === "ArrowUp" && !draft.includes("\n")) {
      if (!history.length) return;
      e.preventDefault();
      const next = cursor < 0 ? history.length - 1 : Math.max(0, cursor - 1);
      setCursor(next);
      setDraft(history[next]);
    } else if (e.key === "ArrowDown" && cursor >= 0) {
      e.preventDefault();
      const next = cursor + 1;
      if (next >= history.length) {
        setCursor(-1);
        setDraft("");
      } else {
        setCursor(next);
        setDraft(history[next]);
      }
    }
  }

  function onInput(e) {
    setDraft(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  }

  const empty = entries.length === 0;

  return (
    <div className="fixed inset-0 bg-black text-gray-300 flex flex-col px-16 py-10 md:px-8 md:py-6 font-mono text-sm leading-relaxed">
      <div
        className="fixed top-3 right-4 z-10 flex items-center gap-4"
        style={{ fontFamily: "inherit" }}
      >
        <Link
          href="/protein-modelling"
          className="text-xs text-emerald-400 hover:text-emerald-300 no-underline font-mono"
        >
          ▶ CKG experiment
        </Link>
        <Link
          href="/tutorials"
          className="text-xs text-green-400 hover:text-green-300 no-underline font-mono"
        >
          ▶ tutorials
        </Link>
        <Link
          href="/pair"
          className="text-xs text-green-400 hover:text-green-300 no-underline font-mono"
        >
          ▶ pair a machine
        </Link>
        {email && (
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>{email}</span>
            <button
              onClick={async () => { await logout(); router.replace("/login"); }}
              className="text-gray-500 hover:text-gray-300 underline"
            >
              log out
            </button>
          </div>
        )}
      </div>
      <div
        ref={historyRef}
        className="flex-1 overflow-y-auto pb-4"
        style={{ scrollbarWidth: "none" }}
      >
        {empty && (
          <div className="mb-8 animate-fade">
            <WelcomePanel />
          </div>
        )}
        {entries.map((e) => (
          <div key={e.id} className="mb-8 animate-fade">
            <div className="text-gray-200 whitespace-pre-wrap mb-2">{e.obs}</div>
            {e.thinking && <span className="text-gray-600 italic">...</span>}
            {e.error && <p className="text-gray-500">[{e.error}]</p>}
            {e.result && <Artifact result={e.result} />}
            {e.multi && (
              <div className="space-y-4">
                {e.multi.map((r, i) => (
                  <div key={i}><Artifact result={r} /></div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-start pt-3">
        <span className="text-gray-700 mr-2 select-none pt-0.5">{proteinsMode ? "🧬" : ""}</span>
        <textarea
          ref={inputRef}
          value={draft}
          onChange={onInput}
          onKeyDown={onKeyDown}
          rows={1}
          spellCheck={false}
          autoFocus
          placeholder={empty ? "type something to store, ask a question, or paste vaHera…" : ""}
          className="flex-1 bg-transparent border-none outline-none resize-none text-gray-200 font-mono text-sm leading-relaxed placeholder-gray-700"
          style={{ caretColor: "#2a9d8f" }}
        />
      </div>

      <style jsx>{`
        .animate-fade {
          animation: fade 0.3s ease;
        }
        @keyframes fade {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        div::-webkit-scrollbar { display: none; }
      `}</style>
    </div>
  );
}
