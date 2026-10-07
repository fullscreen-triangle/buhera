/* ============================================================================
 * Mail — the IMAP half: search, read and sync, one connection per request.
 *
 * Reading never marks a message read: messages are fetched with BODY.PEEK,
 * because opening one on the surface is looking, not acting. Nothing here
 * sends, moves, flags or deletes mail.
 * ========================================================================== */

import fs from "fs";
import path from "path";
import { ImapFlow } from "imapflow";
import { simpleParser } from "mailparser";
import { corpusFile, DEFAULT_LIMIT, gmailRaw, imapSearches, intersect, toMarkdown } from "@/lib/server/mail";

const MAX_TEXT = 200_000;

function client(a) {
  return new ImapFlow({
    host: a.host,
    port: a.port,
    secure: a.secure,
    auth: { user: a.user, pass: a.password },
    logger: false,
    // A self-signed test server (MAIL_ALLOW_SELF_SIGNED=1) only; never for a real account.
    tls: process.env.MAIL_ALLOW_SELF_SIGNED === "1" ? { rejectUnauthorized: false } : undefined,
    connectionTimeout: 20_000,
    greetingTimeout: 15_000,
  });
}

async function withClient(a, fn) {
  const c = client(a);
  await c.connect();
  try {
    return await fn(c);
  } finally {
    await c.logout().catch(() => c.close());
  }
}

const envelopeOf = (account, mailbox, m) => ({
  account: account.id,
  mailbox,
  uid: m.uid,
  date: (m.envelope?.date || m.internalDate)?.toISOString?.() ?? null,
  from: (m.envelope?.from || []).map((x) => ({ name: x.name || "", address: x.address || "" })),
  to: (m.envelope?.to || []).map((x) => ({ name: x.name || "", address: x.address || "" })),
  subject: m.envelope?.subject || "",
  seen: m.flags ? m.flags.has("\\Seen") : null,
  flagged: m.flags ? m.flags.has("\\Flagged") : null,
  size: m.size ?? null,
});

/**
 * Search one account. → { account, messages[], searched: [mailbox], total }
 * `total` counts every match; `messages` holds the newest `limit` of them.
 */
export async function searchAccount(a, q, { limit = DEFAULT_LIMIT } = {}) {
  const mailboxes = q.mailbox ? [q.mailbox] : a.mailboxes;
  return withClient(a, async (c) => {
    const found = [];
    let total = 0;
    for (const mailbox of mailboxes) {
      const lock = await c.getMailboxLock(mailbox, { readOnly: true });
      try {
        let uids;
        if (a.gmail) {
          const raw = gmailRaw(q);
          uids = (await c.search(raw ? { gmraw: raw } : { all: true }, { uid: true })) || [];
        } else {
          const lists = [];
          for (const s of imapSearches(q)) lists.push((await c.search(s, { uid: true })) || []);
          uids = intersect(lists);
        }
        total += uids.length;
        const newest = uids.sort((x, y) => y - x).slice(0, limit);
        if (newest.length) {
          for await (const m of c.fetch(newest, { envelope: true, flags: true, size: true, internalDate: true }, { uid: true })) {
            found.push(envelopeOf(a, mailbox, m));
          }
        }
      } finally {
        lock.release();
      }
    }
    found.sort((x, y) => String(y.date).localeCompare(String(x.date)));
    return { account: a.id, messages: found.slice(0, limit), searched: mailboxes, total };
  });
}

async function parsed(c, uid) {
  const m = await c.fetchOne(String(uid), { source: true, uid: true, envelope: true, flags: true }, { uid: true });
  if (!m?.source) return null;
  const p = await simpleParser(m.source);
  let text = p.text || "";
  const cut = text.length > MAX_TEXT;
  if (cut) text = text.slice(0, MAX_TEXT);
  return {
    uid: m.uid,
    subject: p.subject || "",
    from: p.from?.value || [],
    to: p.to?.value || [],
    cc: p.cc?.value || [],
    date: p.date?.toISOString?.() ?? null,
    messageId: p.messageId || null,
    text,
    truncated: cut,
    attachments: (p.attachments || []).map((x) => ({ filename: x.filename || null, contentType: x.contentType, size: x.size })),
  };
}

/** One message, without marking it read. → message or null */
export async function readMessage(a, mailbox, uid) {
  return withClient(a, async (c) => {
    const lock = await c.getMailboxLock(mailbox, { readOnly: true });
    try {
      const msg = await parsed(c, uid);
      return msg && { account: a.id, mailbox, ...msg };
    } finally {
      lock.release();
    }
  });
}

/** The mailbox of an account whose corpus folder name is `slugged`. */
export async function mailboxForSlug(a, slugged) {
  const slug = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "_";
  const known = a.mailboxes.find((m) => slug(m) === slugged);
  if (known) return known;
  return withClient(a, async (c) => (await c.list()).map((b) => b.path).find((p) => slug(p) === slugged) || null);
}

/**
 * Keep the last `sinceDays` of each mailbox in the corpus, one Markdown file
 * per message. Files already there are left alone, so a second sync only
 * fetches what is new. → { account, written, kept, mailboxes: [{ mailbox, written, kept }] }
 */
export async function syncAccount(a, dir, { sinceDays = 90, max = 2000 } = {}) {
  const since = new Date(Date.now() - sinceDays * 86_400_000);
  return withClient(a, async (c) => {
    const per = [];
    for (const mailbox of a.mailboxes) {
      const lock = await c.getMailboxLock(mailbox, { readOnly: true });
      let written = 0;
      let kept = 0;
      try {
        const uidValidity = String(c.mailbox.uidValidity);
        const uids = ((await c.search({ since }, { uid: true })) || []).sort((x, y) => y - x).slice(0, max);
        // Which are new: the date is in the file name's folder, so look for the uid file anywhere under the mailbox.
        const have = new Set();
        const root = path.dirname(path.dirname(corpusFile(dir, { account: a.id, mailbox, uidValidity, uid: 0, date: 0 })));
        if (fs.existsSync(root)) {
          for (const month of fs.readdirSync(root)) {
            for (const f of fs.readdirSync(path.join(root, month))) have.add(f);
          }
        }
        for (const uid of uids) {
          if (have.has(`${uidValidity}-${uid}.md`)) { kept++; continue; }
          const msg = await parsed(c, uid);
          if (!msg) continue;
          const file = corpusFile(dir, { account: a.id, mailbox, uidValidity, uid, date: msg.date });
          fs.mkdirSync(path.dirname(file), { recursive: true });
          fs.writeFileSync(file, toMarkdown({ ...msg, account: a.id, mailbox }));
          written++;
        }
      } finally {
        lock.release();
      }
      per.push({ mailbox, written, kept });
    }
    return { account: a.id, written: per.reduce((s, m) => s + m.written, 0), kept: per.reduce((s, m) => s + m.kept, 0), mailboxes: per };
  });
}
