/* ============================================================================
 * mail — your accounts, over IMAP (server side: pages/api/mail.js).
 *
 * Search every account at once in the syntax mail clients share (from:,
 * to:, subject:, since:, before:, is:unread, in:, account:, plain words);
 * read a message without marking it read; keep recent mail on disk as
 * Markdown so spraypaint can say, with a coverage verdict, whether your mail
 * says anything about a thing at all. Nothing here sends or changes mail.
 *
 * Instruction shapes:
 *   "accounts" | "show"                         → the accounts and the kept mail
 *   "<query>"                                   → search (same as below)
 *   { kind: "search", query, account?, limit? } → newest matches, every account
 *   { kind: "read", account, mailbox, uid }     → one message
 *   { kind: "open", path }                      → the message a kept-mail passage came from
 *   { kind: "sync", account?, days? }           → keep the last `days` (90) and index them
 *   { kind: "ask", query, dry_run? }            → spraypaint over the kept mail
 * ========================================================================== */

async function post(body) {
  try {
    const res = await fetch("/api/mail", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json().catch(() => null);
    if (!json) return { ok: false, error: `HTTP ${res.status}` };
    return res.ok ? json : { ok: false, ...json, error: json.error || `HTTP ${res.status}` };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });
const failed = (what, error) => done({ kind: "text", lines: [`mail ${what}: ${error}`] }, false);

export const mailModule = {
  id: "mail",

  describe() {
    return {
      id: "mail",
      description:
        "Your mail, over IMAP: search every account at once, read a message without marking it read, " +
        "and keep recent mail on disk so a search can say whether your mail covers a thing at all. " +
        "Never sends, moves or deletes anything.",
      instructions: [
        'mail from:anna since:2026-09-01 protocol',
        'dispatch("mail", { kind: "search", query: "subject:\\"plate layout\\" is:unread" })',
        'dispatch("mail", { kind: "sync", days: 90 })',
        'dispatch("mail", { kind: "ask", query: "internal standard" })',
        'dispatch("mail", "accounts")',
      ],
    };
  },

  async execute(instruction) {
    const inst = typeof instruction === "string"
      ? (["accounts", "show"].includes(instruction.trim()) ? { kind: "accounts" } : { kind: "search", query: instruction })
      : instruction || {};

    switch (inst.kind || "accounts") {
      case "accounts": {
        const r = await post({ action: "accounts" });
        return r.ok === false ? failed("accounts", r.error) : done({ kind: "mail_accounts", ...r });
      }
      case "search": {
        const r = await post({ action: "search", query: inst.query || "", account: inst.account, limit: inst.limit });
        return r.ok === false ? failed("search", r.error) : done({ kind: "mail_search", ...r });
      }
      case "read":
      case "open": {
        const r = await post({ action: inst.kind, account: inst.account, mailbox: inst.mailbox, uid: inst.uid, path: inst.path });
        return r.ok === false ? failed("read", r.error) : done({ kind: "mail_message", message: r.message });
      }
      case "sync": {
        const r = await post({ action: "sync", account: inst.account, days: inst.days });
        return r.ok === false ? failed("sync", r.error) : done({ kind: "mail_sync", ...r });
      }
      case "ask": {
        const r = await post({ action: "ask", query: inst.query, dry_run: inst.dry_run !== false, budget: inst.budget });
        return r.ok === false ? failed("ask", r.error) : done(r.output_delta);
      }
      default:
        return failed("", `unknown kind "${inst.kind}"`);
    }
  },

  outputCell() {
    return { kind: "mail_cell" };
  },
};
