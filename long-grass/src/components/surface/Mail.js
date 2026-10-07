/* ============================================================================
 * Mail — the accounts page, a search's messages, one message, a sync.
 *
 * Every action starts a new frame (useStep): searching, opening a message,
 * syncing. A message can be kept on a plan from wherever it is shown.
 * ========================================================================== */

import { useState } from "react";
import { useStep } from "@/components/surface/actions";
import { Button, Row } from "@/components/surface/controls";
import { KeepOnPlan } from "@/components/surface/Planning";

const EXAMPLE = `{
  "dir": "~/.buhera/mail",
  "accounts": [
    { "id": "uni", "label": "university",
      "host": "<your provider's IMAP host>", "user": "<login>",
      "password_env": "MAIL_UNI_PASSWORD" },
    { "id": "gmail", "host": "imap.gmail.com", "user": "you@gmail.com",
      "password_env": "MAIL_GMAIL_PASSWORD" }
  ]
}`;

const who = (list) => (list || []).map((a) => a.name || a.address).filter(Boolean).join(", ");
const day = (iso) => (iso ? iso.slice(0, 10) : "");

function SearchBox({ placeholder = "from:anna since:2026-09-01 protocol" }) {
  const step = useStep();
  const [q, setQ] = useState("");
  if (!step) return null;
  const go = () => step(`mail ${q}`, "mail", { kind: "search", query: q });
  return (
    <div className="flex gap-2 items-center">
      <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") go(); }}
        placeholder={placeholder} spellCheck={false}
        className="flex-1 bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-sm text-gray-200 px-1" />
      <Button onClick={go}>search</Button>
    </div>
  );
}

export function MailAccounts({ file, exists, dir, indexed, problems = [], accounts = [] }) {
  const step = useStep();
  const [days, setDays] = useState(90);
  const ready = accounts.filter((a) => a.ready);
  return (
    <div>
      {!exists && (
        <div className="mb-6">
          <p className="text-gray-300 mb-2">No mail accounts yet. Put them in <span className="font-mono text-teal-300/80 break-all">{file}</span>:</p>
          <pre className="p-3 rounded border border-gray-900 bg-white/[0.02] text-xs font-mono text-gray-300 whitespace-pre-wrap">{EXAMPLE}</pre>
          <p className="text-xs text-gray-500 mt-2 leading-relaxed">
            Each password stays in a variable on this machine (in <span className="font-mono">.env.local</span>, beside the other keys), named by
            <span className="font-mono"> password_env</span>. Gmail needs an app password (Google account → Security → App passwords).
            Then open this page again.
          </p>
        </div>
      )}
      {problems.map((p, i) => <p key={i} className="text-xs text-rose-300/80">{p}</p>)}

      {accounts.map((a) => (
        <Row key={a.id} label={a.label} hint={`${a.user} @ ${a.host}`}>
          {a.ready
            ? <span className="text-xs text-gray-400">ready · {a.mailboxes.join(", ")}{a.gmail ? " · Gmail search syntax" : ""}</span>
            : <span className="text-xs text-rose-300/80">{a.problem}</span>}
        </Row>
      ))}

      {ready.length > 0 && (
        <>
          <div className="mt-6 mb-2 text-[11px] uppercase tracking-wider text-gray-600">search every account</div>
          <SearchBox />
          <p className="text-[11px] text-gray-600 mt-1">
            from: to: cc: subject: since:YYYY-MM-DD before: is:unread is:flagged in:Mailbox account:id — and any words.
            Or write <span className="text-gray-400">mail …</span> on the blank screen.
          </p>

          <div className="mt-6 mb-2 text-[11px] uppercase tracking-wider text-gray-600">kept mail</div>
          <p className="text-xs text-gray-500 mb-2 leading-relaxed">
            Keeping your recent mail on this machine ({dir}) lets a search say whether your mail covers a thing at all —
            <span className="text-gray-300"> covered</span>, <span className="text-gray-300">partial</span> or{" "}
            <span className="text-gray-300">declined</span> — instead of returning its best look-alike. {indexed ? "Kept mail is indexed." : "Nothing is kept yet."}
          </p>
          {step && (
            <span className="inline-flex items-center gap-2 text-xs text-gray-500">
              keep the last
              <input type="number" min={1} max={3650} value={days} onChange={(e) => setDays(Number(e.target.value) || 90)}
                className="w-16 bg-transparent border-b border-gray-800 text-gray-200 outline-none" />
              days
              <Button onClick={() => step(`mail sync ${days} days`, "mail", { kind: "sync", days })}>{indexed ? "sync again" : "sync"}</Button>
            </span>
          )}
        </>
      )}
    </div>
  );
}

export function MailRow({ m, query }) {
  const step = useStep();
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 py-1.5 border-t border-gray-900">
      <span className="text-[11px] text-gray-600 w-20 shrink-0">{day(m.date)}</span>
      {m.seen === false && <span className="w-1.5 h-1.5 rounded-full bg-teal-400 self-center" title="unread" />}
      <button type="button" className={`text-sm text-left hover:text-white ${m.seen === false ? "text-gray-100" : "text-gray-300"}`}
        onClick={() => step?.(`mail: ${m.subject}`, "mail", { kind: "read", account: m.account, mailbox: m.mailbox, uid: m.uid })}>
        {m.subject || "(no subject)"}
      </button>
      <span className="text-[11px] text-gray-500">{who(m.from)}</span>
      <span className="text-[11px] text-gray-700">{m.account}{m.mailbox !== "INBOX" ? ` · ${m.mailbox}` : ""}</span>
      <KeepOnPlan found={{ source: "mail", cite: `mail:${m.account}/${m.mailbox}/${m.uid}`, title: m.subject, snippet: `from ${who(m.from)}, ${day(m.date)}`, query, mail: { account: m.account, mailbox: m.mailbox, uid: m.uid } }} />
    </div>
  );
}

export function MailSearch({ query, problems = [], accounts = [], messages = [] }) {
  const total = accounts.reduce((s, a) => s + (a.total || 0), 0);
  return (
    <div>
      <div className="text-xs text-gray-500 mb-3">
        mail <span className="text-white">“{query || "everything"}”</span> · {total} match{total === 1 ? "" : "es"}
        {messages.length < total ? `, the newest ${messages.length} shown` : ""}
        {" · "}{accounts.map((a) => `${a.account} ${a.error ? "failed" : a.total}`).join(" · ")}
      </div>
      {problems.map((p, i) => <p key={i} className="text-xs text-amber-300/80">{p}</p>)}
      {accounts.filter((a) => a.error).map((a) => <p key={a.account} className="text-xs text-rose-300/80">{a.account}: {a.error}</p>)}
      {messages.length === 0 && <p className="text-gray-500">no message matches.</p>}
      {messages.map((m) => <MailRow key={`${m.account}/${m.mailbox}/${m.uid}`} m={m} query={query} />)}
      <div className="mt-6"><SearchBox placeholder="search again…" /></div>
    </div>
  );
}

export function MailMessage({ message: m }) {
  if (!m) return <p className="text-gray-500">(no message)</p>;
  return (
    <div>
      <div className="text-lg text-gray-100 mb-2">{m.subject || "(no subject)"}</div>
      <div className="text-xs text-gray-500 space-y-0.5 mb-4">
        <div><span className="text-gray-600 w-10 inline-block">from</span> {who(m.from)} <span className="text-gray-700">{m.from?.[0]?.address}</span></div>
        {m.to?.length > 0 && <div><span className="text-gray-600 w-10 inline-block">to</span> {who(m.to)}</div>}
        {m.cc?.length > 0 && <div><span className="text-gray-600 w-10 inline-block">cc</span> {who(m.cc)}</div>}
        <div><span className="text-gray-600 w-10 inline-block">date</span> {m.date ? new Date(m.date).toLocaleString() : "—"} <span className="text-gray-700">· {m.account} / {m.mailbox} · uid {m.uid}</span></div>
        {m.attachments?.length > 0 && (
          <div><span className="text-gray-600 w-10 inline-block">files</span> {m.attachments.map((a) => `${a.filename || "(unnamed)"} (${Math.max(1, Math.round((a.size || 0) / 1024))} KB)`).join(", ")}</div>
        )}
      </div>
      <div className="mb-4">
        <KeepOnPlan found={{ source: "mail", cite: `mail:${m.account}/${m.mailbox}/${m.uid}`, title: m.subject, snippet: String(m.text || "").slice(0, 400), mail: { account: m.account, mailbox: m.mailbox, uid: m.uid } }} />
      </div>
      <div className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed border-t border-gray-900 pt-4">{m.text || "(no text)"}</div>
      {m.truncated && <p className="text-[11px] text-gray-600 mt-2">(cut at 200,000 characters)</p>}
    </div>
  );
}

export function MailSync({ dir, days, synced = [], index }) {
  return (
    <div className="text-sm">
      <div className="text-xs text-gray-500 mb-3">kept the last {days} days of mail in <span className="font-mono">{dir}</span></div>
      {synced.map((s) => (
        <div key={s.account} className="py-1 border-t border-gray-900">
          <span className="text-gray-200">{s.account}</span>{" "}
          {s.error
            ? <span className="text-rose-300/80 text-xs">{s.error}</span>
            : <span className="text-xs text-gray-400">{s.written} new, {s.kept} already kept{s.mailboxes?.length > 1 ? ` (${s.mailboxes.map((b) => `${b.mailbox} ${b.written}`).join(", ")})` : ""}</span>}
        </div>
      ))}
      <p className="text-xs mt-3">
        {index?.ok
          ? <span className="text-gray-400">indexed {index.documents} message{index.documents === 1 ? "" : "s"} — searches now answer with a verdict.</span>
          : <span className="text-rose-300/80">not indexed: {index?.error}</span>}
      </p>
    </div>
  );
}
