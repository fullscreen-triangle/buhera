// API route for mail: your accounts, over IMAP, read in place.
//
//   POST /api/mail { action: "accounts" }                        → which accounts, and whether each can connect
//   POST /api/mail { action: "search", query, account?, limit? } → the newest matching messages, every account
//   POST /api/mail { action: "read", account, mailbox, uid }     → one message (never marked read)
//   POST /api/mail { action: "open", path }                      → the message a corpus passage came from
//   POST /api/mail { action: "sync", account?, days? }           → keep recent mail as Markdown, then index it
//   POST /api/mail { action: "ask", query, dry_run?, budget? }   → spraypaint over the kept mail, with its verdict
//
// Mail is answered only for requests from the machine the server runs on
// (isLocalRequest): this server has no way to know whose mail a remote
// visitor may read. Accounts and the corpus folder: lib/server/mail.js.

import fs from "fs";
import path from "path";
import { isLocalRequest } from "@/lib/server/rag";
import { configPath, parseConfig, parseMailQuery, publicAccount, refFromPath } from "@/lib/server/mail";
import { mailboxForSlug, readMessage, searchAccount, syncAccount } from "@/lib/server/mail-imap";
import { findBinary, run } from "@/lib/server/spawn";
import { askArgs, parseJsonLoose } from "@/lib/server/spraypaint";

function config() {
  const file = configPath();
  let raw = null;
  try { raw = fs.readFileSync(file, "utf8"); } catch { /* no file yet */ }
  return { file, exists: raw != null, ...parseConfig(raw) };
}

const indexed = (dir) => fs.existsSync(path.join(dir, ".spraypaint", "index.json"));

async function indexCorpus(dir) {
  const bin = findBinary("spraypaint", "SPRAYPAINT_CLI");
  if (!bin) return { ok: false, error: "spraypaint is not installed, so the kept mail cannot be indexed" };
  // .spraypaint/ pins the corpus root here, whatever lies above it.
  fs.mkdirSync(path.join(dir, ".spraypaint"), { recursive: true });
  const r = await run(bin, ["index", "--root", dir, "--json"], { timeoutMs: 600_000 });
  if (r.code !== 0) return { ok: false, error: `spraypaint index exited with ${r.code}: ${r.stderr.trim().slice(0, 300)}` };
  return { ok: true, ...(parseJsonLoose(r.stdout) || {}) };
}

const why = (e) => String(e?.responseText || e?.message || e);

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "method not allowed" });
  if (!isLocalRequest(req)) {
    return res.status(403).json({ ok: false, error: "mail is read only on the machine this server runs on — run long-grass locally to reach your accounts" });
  }

  const { action = "accounts", query, account, mailbox, uid, limit, days, path: rel, dry_run, budget } = req.body ?? {};
  const cfg = config();
  const pick = (id) => cfg.accounts.filter((a) => (!id || a.id === id));

  if (action === "accounts") {
    return res.status(200).json({
      ok: true,
      file: cfg.file,
      exists: cfg.exists,
      dir: cfg.dir,
      indexed: indexed(cfg.dir),
      problems: cfg.problems,
      accounts: cfg.accounts.map(publicAccount),
    });
  }

  if (action === "search") {
    const q = parseMailQuery(query);
    const targets = pick(account || q.account);
    if (!targets.length) return res.status(404).json({ ok: false, error: cfg.accounts.length ? `no account "${account || q.account}"` : `no accounts yet — add them to ${cfg.file}` });
    const results = await Promise.all(
      targets.map(async (a) => {
        if (!a.ready) return { account: a.id, error: a.problem, messages: [] };
        try { return await searchAccount(a, q, { limit: Number(limit) || undefined }); } catch (e) { return { account: a.id, error: why(e), messages: [] }; }
      })
    );
    const messages = results.flatMap((r) => r.messages).sort((x, y) => String(y.date).localeCompare(String(x.date)));
    return res.status(200).json({ ok: true, query, problems: q.problems, accounts: results.map(({ messages: m, ...r }) => ({ ...r, count: m.length })), messages });
  }

  if (action === "read" || action === "open") {
    let target = { account, mailbox, uid };
    if (action === "open") {
      const ref = refFromPath(rel);
      if (!ref) return res.status(400).json({ ok: false, error: `"${rel}" is not a path in the kept mail` });
      const a = pick(ref.account)[0];
      if (!a) return res.status(404).json({ ok: false, error: `no account "${ref.account}"` });
      target = { account: a.id, mailbox: await mailboxForSlug(a, ref.mailboxSlug).catch(() => null), uid: ref.uid };
      if (!target.mailbox) return res.status(404).json({ ok: false, error: `account ${a.id} has no mailbox matching "${ref.mailboxSlug}"` });
    }
    const a = pick(target.account)[0];
    if (!a) return res.status(404).json({ ok: false, error: `no account "${target.account}"` });
    if (!a.ready) return res.status(409).json({ ok: false, error: a.problem });
    try {
      const msg = await readMessage(a, target.mailbox, Number(target.uid));
      if (!msg) return res.status(404).json({ ok: false, error: `no message ${target.uid} in ${target.mailbox}` });
      return res.status(200).json({ ok: true, message: msg });
    } catch (e) {
      return res.status(502).json({ ok: false, error: why(e) });
    }
  }

  if (action === "sync") {
    const targets = pick(account).filter((a) => a.ready);
    if (!targets.length) return res.status(404).json({ ok: false, error: "no account is ready to sync" });
    const synced = [];
    for (const a of targets) {
      try { synced.push(await syncAccount(a, cfg.dir, { sinceDays: Number(days) || 90 })); } catch (e) { synced.push({ account: a.id, error: why(e) }); }
    }
    const index = await indexCorpus(cfg.dir);
    return res.status(200).json({ ok: true, dir: cfg.dir, days: Number(days) || 90, synced, index });
  }

  if (action === "ask") {
    if (typeof query !== "string" || !query.trim()) return res.status(400).json({ ok: false, error: "query is required" });
    if (!indexed(cfg.dir)) return res.status(409).json({ ok: false, error: "the kept mail is not indexed yet — sync first" });
    const bin = findBinary("spraypaint", "SPRAYPAINT_CLI");
    if (!bin) return res.status(503).json({ ok: false, error: "spraypaint is not installed" });
    const r = await run(bin, askArgs({ query, root: cfg.dir, budget, dryRun: !!dry_run }), { timeoutMs: 300_000 });
    if (r.code !== 0) return res.status(502).json({ ok: false, error: `spraypaint exited with ${r.code}`, stderr: r.stderr.trim() });
    const parsed = parseJsonLoose(r.stdout);
    if (!parsed) return res.status(502).json({ ok: false, error: "could not read spraypaint's output" });
    return res.status(200).json({ ok: true, output_delta: { kind: "spraypaint_result", query, corpus: "mail", elapsed_ms: r.elapsed_ms, ...parsed } });
  }

  return res.status(400).json({ ok: false, error: `unknown action "${action}"` });
}
