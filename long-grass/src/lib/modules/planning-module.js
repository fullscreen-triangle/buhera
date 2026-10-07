/* ============================================================================
 * planning — find things, and plan what to do with them.
 *
 * `find` asks every place a thing might be, at once, and shows each answer
 * under its source, with that source's own honesty:
 *
 *   mail    your accounts, searched live over IMAP; and, once mail has been
 *           kept (mail sync), spraypaint over it, with a coverage verdict
 *   files   spraypaint over this machine's tree, with a coverage verdict
 *   web     a model with a search tool (needs GEMINI_API_KEY on the server)
 *   plans   your own plan items that mention it
 *
 * Every hit can be kept on a plan item (lib/surface/planning.js) together
 * with where it came from and the verdict of the search that found it.
 * Searches are previews (dry runs): they commit nothing to spraypaint's
 * count. Keeping a hit on a plan is relying on it, and commits the ask then.
 *
 * Instruction shapes:
 *   "show" | "board"                                 → the plan board
 *   "<words>"                                         → find
 *   { kind: "find", query, sources? }                 → find in ["mail","files","web","plans"]
 *   { kind: "new", title, type?: "experiment"|"task", due? }
 *   { kind: "item", id }
 * ========================================================================== */

import { addItem, getItems, matchItems } from "@/lib/surface/planning";

export const SOURCES = ["mail", "files", "web", "plans"];

async function post(path, body) {
  try {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!json) return { ok: false, error: `HTTP ${res.status}` };
    return res.ok ? { ok: true, ...json } : { ok: false, ...json, error: json.error || `HTTP ${res.status}` };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

const FINDERS = {
  async mail(query) {
    const accounts = await post("/api/mail", { action: "accounts" });
    if (!accounts.ok) return { ok: false, error: accounts.error };
    if (!accounts.accounts?.length) return { ok: false, error: `no mail accounts yet — add them to ${accounts.file}` };
    const [live, kept] = await Promise.all([
      post("/api/mail", { action: "search", query, limit: 8 }),
      accounts.indexed ? post("/api/mail", { action: "ask", query, dry_run: true, budget: 6 }) : Promise.resolve(null),
    ]);
    return {
      ok: live.ok || !!kept?.ok,
      error: live.ok ? null : live.error,
      live: live.ok ? { messages: live.messages, accounts: live.accounts, problems: live.problems } : null,
      kept: kept?.ok ? kept.output_delta : null,
      keptNote: accounts.indexed ? (kept && !kept.ok ? kept.error : null) : "no kept mail yet, so no verdict — sync mail to get one",
    };
  },
  async files(query) {
    const r = await post("/api/spraypaint", { action: "ask", query, dry_run: true, budget: 6 });
    return r.ok ? { ok: true, result: r.output_delta } : { ok: false, error: r.error };
  },
  async web(query) {
    const r = await post("/api/web-search", { query });
    return r.ok ? { ok: true, result: r.output_delta } : { ok: false, error: r.error };
  },
  async plans(query) {
    const hits = matchItems(query, getItems());
    return { ok: true, items: hits.map((i) => ({ id: i.id, title: i.title, kind: i.kind, status: i.status })) };
  },
};

export async function find(query, sources = SOURCES) {
  const wanted = sources.filter((s) => FINDERS[s]);
  const sections = await Promise.all(
    wanted.map(async (source) => {
      try { return { source, ...(await FINDERS[source](query)) }; } catch (err) { return { source, ok: false, error: err.message || String(err) }; }
    })
  );
  return { kind: "planning_find", query, sections };
}

const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });

export const planningModule = {
  id: "planning",

  describe() {
    return {
      id: "planning",
      description:
        "Find things — in your mail, your files, on the web — and plan what to do with them: experiments and tasks, " +
        "with steps, the things you found (and how sure each search was), and the AppHub jobs that run them.",
      instructions: [
        "find plate layout lipid",
        "plan experiment PC 34:1 lipid series on LARA",
        "plan task book the RTX 4090 session",
        'dispatch("planning", { kind: "find", query: "internal standard", sources: ["mail", "files"] })',
      ],
    };
  },

  async execute(instruction) {
    const inst = typeof instruction === "string"
      ? (["show", "board", ""].includes(instruction.trim()) ? { kind: "board" } : { kind: "find", query: instruction })
      : instruction || {};
    switch (inst.kind || "board") {
      case "board":
      case "show":
        return done({ kind: "planning_board" });
      case "find": {
        const q = String(inst.query || "").trim();
        if (!q) return done({ kind: "text", lines: ["find: what are you looking for?"] }, false);
        return done(await find(q, Array.isArray(inst.sources) && inst.sources.length ? inst.sources : SOURCES));
      }
      case "new": {
        const item = addItem({ title: inst.title, kind: inst.type || inst.itemKind || "task", due: inst.due || null });
        return done({ kind: "planning_item", id: item.id, created: true });
      }
      case "item":
        return done({ kind: "planning_item", id: inst.id });
      default:
        return done({ kind: "text", lines: [`planning: unknown kind "${inst.kind}"`] }, false);
    }
  },

  outputCell() {
    return { kind: "planning_cell" };
  },
};
