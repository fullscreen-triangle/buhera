// Tests for mail's pure half: the accounts file, the query syntax, the IMAP
// searches and Gmail syntax it becomes, and the kept-mail corpus format.
import test from "node:test";
import assert from "node:assert/strict";

import { corpusFile, gmailRaw, imapSearches, intersect, parseConfig, parseMailQuery, publicAccount, refFromPath, toMarkdown } from "../src/lib/server/mail.js";

test("accounts: defaults, Gmail by host, and why an account is not ready", () => {
  const cfg = parseConfig(JSON.stringify({ accounts: [
    { id: "uni", host: "imap.example.org", user: "me", password_env: "PW_UNI" },
    { id: "gmail", host: "imap.gmail.com", user: "me@gmail.com", password_env: "PW_G" },
    { id: "bare", host: "x" },
  ] }), { PW_UNI: "secret" });
  const [uni, gmail, bare] = cfg.accounts;
  assert.equal(uni.ready, true);
  assert.equal(uni.port, 993);
  assert.equal(uni.secure, true);
  assert.deepEqual(uni.mailboxes, ["INBOX"]);
  assert.equal(gmail.gmail, true);
  assert.deepEqual(gmail.mailboxes, ["[Gmail]/All Mail"]);
  assert.match(gmail.problem, /PW_G is not set/);
  assert.equal(bare.problem, "no user");
});

test("no password ever reaches the browser", () => {
  const a = parseConfig(JSON.stringify({ accounts: [{ host: "h", user: "u", password: "hunter2" }] })).accounts[0];
  assert.equal(a.password, "hunter2");
  assert.equal(JSON.stringify(publicAccount(a)).includes("hunter2"), false);
});

test("a missing or broken accounts file is reported, not thrown", () => {
  assert.deepEqual(parseConfig(null).accounts, []);
  assert.match(parseConfig("{nope").problems[0], /not valid JSON/);
});

test("the query syntax", () => {
  const q = parseMailQuery('from:anna subject:"sample prep" since:2026-09-01 is:unread in:Sent internal standard foo:bar');
  assert.deepEqual(q.from, ["anna"]);
  assert.deepEqual(q.subject, ["sample prep"]);
  assert.equal(q.since, "2026-09-01");
  assert.equal(q.unseen, true);
  assert.equal(q.mailbox, "Sent");
  assert.deepEqual(q.words, ["internal", "standard", "foo:bar"]);
  assert.match(parseMailQuery("since:yesterday").problems[0], /YYYY-MM-DD/);
});

test("every term is its own IMAP search; addresses by header, dates by when written", () => {
  const s = imapSearches(parseMailQuery("from:mark since:2026-09-29 lara"));
  assert.equal(s.length, 2);
  assert.deepEqual(s[0].header, { from: "mark" });
  assert.equal(s[1].text, "lara");
  assert.ok(s.every((x) => x.sentSince instanceof Date));
  assert.deepEqual(imapSearches(parseMailQuery("")), [{ all: true }]);
  assert.deepEqual(intersect([[1, 2, 5], [5, 1], [1, 5, 9]]).sort(), [1, 5]);
});

test("Gmail gets its own syntax", () => {
  assert.equal(gmailRaw(parseMailQuery('from:anna subject:"plate layout" since:2026-09-01 is:unread lipid')),
    'from:anna subject:"plate layout" lipid after:2026/09/01 is:unread');
});

test("a kept message's path leads back to it", () => {
  const f = corpusFile("/m", { account: "uni", mailbox: "[Gmail]/All Mail", uidValidity: 77, uid: 12, date: "2026-09-22T12:30:00Z" });
  const rel = f.replace(/\\/g, "/").replace(/^\/m\//, "");
  assert.equal(rel, "uni/Gmail_All_Mail/2026-09/77-12.md");
  assert.deepEqual(refFromPath(rel), { account: "uni", mailboxSlug: "Gmail_All_Mail", uidValidity: 77, uid: 12 });
  assert.equal(refFromPath("notes/readme.md"), null);
});

test("a kept message is Markdown with its headers searchable", () => {
  const md = toMarkdown({ subject: "Sample prep", from: [{ name: "Anna", address: "anna@uni.test" }], to: [{ address: "me@uni.test" }],
    date: "2026-09-22T12:30:00Z", account: "uni", mailbox: "INBOX", uid: 2, text: "MTBE\r\nextraction", attachments: [{ filename: "plate.csv" }] });
  assert.match(md, /^# Sample prep\n/);
  assert.match(md, /from: Anna <anna@uni.test>/);
  assert.match(md, /date: 2026-09-22 12:30 UTC/);
  assert.match(md, /attachments: plate.csv/);
  assert.match(md, /MTBE\nextraction/);
});
