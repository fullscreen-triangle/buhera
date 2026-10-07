/* ============================================================================
 * Mail — the pure half: which accounts exist, what a query asks for, and how
 * a message is kept on disk. The IMAP half is mail-imap.js.
 *
 * Accounts live in a file outside any repository, MAIL_ACCOUNTS_FILE or
 * ~/.buhera/mail.json:
 *
 *   {
 *     "dir": "~/.buhera/mail",
 *     "accounts": [
 *       { "id": "uni", "label": "university", "host": "imap.example.org",
 *         "user": "me", "password_env": "MAIL_UNI_PASSWORD" },
 *       { "id": "gmail", "host": "imap.gmail.com", "user": "me@gmail.com",
 *         "password_env": "MAIL_GMAIL_PASSWORD" }
 *     ]
 *   }
 *
 * `port` defaults to 993 and `secure` to true. A password is named by
 * `password_env` (the variable holding it) or, less safely, given as
 * `password`. No password ever leaves the server: publicAccount() is what a
 * browser sees. Gmail is detected by host (or `"gmail": true`) and searched
 * with Gmail's own syntax.
 *
 * The corpus. `sync` keeps each message as one Markdown file,
 *   <dir>/<account>/<mailbox>/<YYYY-MM>/<uidvalidity>-<uid>.md
 * so spraypaint can index the folder and answer, with a coverage verdict,
 * whether your mail says anything about a thing at all. refFromPath() turns
 * a passage's path back into the message it came from.
 * ========================================================================== */

import os from "os";
import path from "path";

export const DEFAULT_LIMIT = 30;

const expandHome = (p) => (p && p.startsWith("~") ? path.join(os.homedir(), p.slice(1)) : p);

export function configPath(env = process.env) {
  return expandHome(env.MAIL_ACCOUNTS_FILE || "~/.buhera/mail.json");
}

const isGmailHost = (host) => /(^|\.)gmail\.com$|googlemail\.com$/i.test(String(host || ""));

/**
 * Read the accounts file's text into accounts. Each account says whether it
 * is ready to connect and, when not, why.
 * → { dir, accounts: [{ id, label, host, port, secure, user, gmail, mailboxes, password, ready, problem }], problems }
 */
export function parseConfig(raw, env = process.env) {
  if (raw == null) return { dir: expandHome("~/.buhera/mail"), accounts: [], problems: [] };
  let doc;
  try {
    doc = JSON.parse(raw);
  } catch (e) {
    return { dir: expandHome("~/.buhera/mail"), accounts: [], problems: [`the accounts file is not valid JSON: ${e.message}`] };
  }
  const seen = new Set();
  const accounts = (Array.isArray(doc.accounts) ? doc.accounts : []).map((a, i) => {
    const id = String(a.id || `account${i + 1}`).replace(/[^A-Za-z0-9._-]/g, "_");
    const gmail = a.gmail ?? isGmailHost(a.host);
    const password = a.password_env ? env[a.password_env] : a.password;
    let problem = null;
    if (seen.has(id)) problem = `the id "${id}" is used twice`;
    else if (!a.host) problem = "no host";
    else if (!a.user) problem = "no user";
    else if (a.password_env && !password) problem = `the variable ${a.password_env} is not set on this server`;
    else if (!password) problem = "no password_env (or password)";
    seen.add(id);
    return {
      id,
      label: a.label || id,
      host: a.host,
      port: Number(a.port) || 993,
      secure: a.secure !== false,
      user: a.user,
      gmail,
      mailboxes: Array.isArray(a.mailboxes) && a.mailboxes.length ? a.mailboxes : [gmail ? "[Gmail]/All Mail" : "INBOX"],
      password,
      ready: !problem,
      problem,
    };
  });
  return { dir: expandHome(doc.dir || "~/.buhera/mail"), accounts, problems: [] };
}

/** An account as a browser may see it: no password, no server detail it does not need. */
export function publicAccount(a) {
  return { id: a.id, label: a.label, host: a.host, user: a.user, gmail: a.gmail, mailboxes: a.mailboxes, ready: a.ready, problem: a.problem };
}

// ── queries ──────────────────────────────────────────────────────────────

const KEYS = new Set(["from", "to", "cc", "subject", "since", "after", "before", "in", "account", "is"]);
const DATE = /^\d{4}-\d{2}-\d{2}$/;

function tokens(q) {
  const out = [];
  const re = /(\w+):"([^"]*)"|(\w+):(\S+)|"([^"]*)"|(\S+)/g;
  let m;
  while ((m = re.exec(q))) {
    if (m[1] || m[3]) out.push({ key: (m[1] || m[3]).toLowerCase(), value: m[2] ?? m[4] });
    else out.push({ value: m[5] ?? m[6], phrase: m[5] != null });
  }
  return out;
}

/**
 * A mail query, in the syntax most mail clients share:
 *   from:anna  to:me  subject:"sample prep"  since:2026-09-01  before:2026-10-01
 *   is:unread  is:flagged  in:Sent  account:uni   and any other words
 * Words must all occur (in the subject, the addresses or the body).
 * A key this parser does not know is kept as a word, so `foo:bar` is not lost.
 */
export function parseMailQuery(query) {
  const q = { from: [], to: [], cc: [], subject: [], words: [], since: null, before: null, unseen: false, seen: false, flagged: false, mailbox: null, account: null, problems: [] };
  for (const t of tokens(String(query || ""))) {
    if (!t.key || !KEYS.has(t.key)) {
      if (t.key) q.words.push(`${t.key}:${t.value}`);
      else if (t.value) q.words.push(t.value);
      continue;
    }
    const v = t.value;
    if (t.key === "since" || t.key === "after" || t.key === "before") {
      if (!DATE.test(v)) { q.problems.push(`${t.key}: needs a date as YYYY-MM-DD, got "${v}"`); continue; }
      if (t.key === "before") q.before = v; else q.since = v;
    } else if (t.key === "is") {
      const w = v.toLowerCase();
      if (w === "unread") q.unseen = true;
      else if (w === "read") q.seen = true;
      else if (w === "flagged" || w === "starred") q.flagged = true;
      else q.problems.push(`is:${v} is not known (unread, read, flagged)`);
    } else if (t.key === "in") q.mailbox = v;
    else if (t.key === "account") q.account = v;
    else q[t.key].push(v);
  }
  return q;
}

/**
 * The IMAP searches whose results, intersected, answer the query. IMAP
 * SEARCH takes each key once per command, so every term is its own search;
 * dates and flags ride along on each one. Addresses use HEADER FROM (a
 * substring match everywhere; plain FROM is exact on some servers), and dates
 * the date the mail was written (SENTSINCE), not when it arrived.
 */
export function imapSearches(q) {
  const base = {};
  if (q.since) base.sentSince = new Date(`${q.since}T00:00:00Z`);
  if (q.before) base.sentBefore = new Date(`${q.before}T00:00:00Z`);
  if (q.unseen) base.seen = false;
  if (q.seen) base.seen = true;
  if (q.flagged) base.flagged = true;
  const terms = [
    ...q.from.map((v) => ({ header: { from: v } })),
    ...q.to.map((v) => ({ header: { to: v } })),
    ...q.cc.map((v) => ({ header: { cc: v } })),
    ...q.subject.map((v) => ({ subject: v })),
    ...q.words.map((v) => ({ text: v })),
  ];
  if (!terms.length) return [Object.keys(base).length ? base : { all: true }];
  return terms.map((t) => ({ ...base, ...t }));
}

/** The same query in Gmail's own search syntax (X-GM-RAW). */
export function gmailRaw(q) {
  const quote = (v) => (/\s/.test(v) ? `"${v}"` : v);
  const parts = [
    ...q.from.map((v) => `from:${quote(v)}`),
    ...q.to.map((v) => `to:${quote(v)}`),
    ...q.cc.map((v) => `cc:${quote(v)}`),
    ...q.subject.map((v) => `subject:${quote(v)}`),
    ...q.words.map(quote),
  ];
  if (q.since) parts.push(`after:${q.since.replace(/-/g, "/")}`);
  if (q.before) parts.push(`before:${q.before.replace(/-/g, "/")}`);
  if (q.unseen) parts.push("is:unread");
  if (q.seen) parts.push("is:read");
  if (q.flagged) parts.push("is:starred");
  return parts.join(" ");
}

export function intersect(lists) {
  if (!lists.length) return [];
  let acc = new Set(lists[0]);
  for (const l of lists.slice(1)) {
    const s = new Set(l);
    acc = new Set([...acc].filter((x) => s.has(x)));
  }
  return [...acc];
}

// ── the corpus ───────────────────────────────────────────────────────────

const slug = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "_";

export function corpusFile(dir, { account, mailbox, uidValidity, uid, date }) {
  const d = date ? new Date(date) : new Date(0);
  const month = Number.isNaN(d.getTime()) ? "unknown" : d.toISOString().slice(0, 7);
  return path.join(dir, slug(account), slug(mailbox), month, `${uidValidity}-${uid}.md`);
}

/** A corpus path (as spraypaint reports it, `/`-separated) back to its message. */
export function refFromPath(rel) {
  const m = /^([^/]+)\/([^/]+)\/[^/]+\/(\d+)-(\d+)\.md$/.exec(String(rel).replace(/\\/g, "/"));
  if (!m) return null;
  return { account: m[1], mailboxSlug: m[2], uidValidity: Number(m[3]), uid: Number(m[4]) };
}

const addr = (list) => (list || []).map((a) => (a.name ? `${a.name} <${a.address}>` : a.address)).join(", ");

/** One message as the Markdown file the corpus keeps. */
export function toMarkdown({ subject, from, to, cc, date, account, mailbox, uid, text, attachments }) {
  const lines = [`# ${subject || "(no subject)"}`, ""];
  lines.push(`from: ${addr(from)}`);
  if (to?.length) lines.push(`to: ${addr(to)}`);
  if (cc?.length) lines.push(`cc: ${addr(cc)}`);
  if (date) lines.push(`date: ${new Date(date).toISOString().replace("T", " ").slice(0, 16)} UTC`);
  lines.push(`mailbox: ${account} / ${mailbox} · uid ${uid}`);
  if (attachments?.length) lines.push(`attachments: ${attachments.map((a) => a.filename || "(unnamed)").join(", ")}`);
  lines.push("", String(text || "").replace(/\r\n/g, "\n").trim(), "");
  return lines.join("\n");
}
