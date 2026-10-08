// Tests for the landing document: the Markdown format of prose, cells and
// their outputs; the command line; who may run it; scripts and file search.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import { command, infoTimeout, parse, serialize, STARTER } from "../src/lib/notebook/document.js";
import { appendRecord, owner, readRecord, runScript, scriptOutput, searchFiles, tokenSubject } from "../src/lib/server/notebook.js";
import { pickProvider } from "../src/lib/server/notebook-ask.js";

test("a cell's output block belongs to the cell above it, and survives a round trip", () => {
  const md = "# T\n\nsome prose\n\n```ask\nwhat is DCAT?\n```\n\n```output\nan answer\n```\n\n```python timeout=5\nprint(1)\n```\n";
  const b = parse(md);
  assert.deepEqual(b.map((x) => x.type), ["prose", "cell", "cell"]);
  assert.equal(b[1].kind, "ask");
  assert.equal(b[1].output, "an answer");
  assert.equal(b[2].output, null);
  assert.equal(b[2].info, "timeout=5");
  assert.equal(infoTimeout(b[2].info), 5);
  assert.deepEqual(parse(serialize(b)), b);
});

test("an output containing a fence is written in a longer fence and read back whole", () => {
  const out = "here:\n\n```python\nprint(2)\n```\n\ndone";
  const md = serialize([{ type: "cell", kind: "ask", info: "", source: "q", output: out }]);
  assert.match(md, /````output/);
  const [c] = parse(md);
  assert.equal(c.output, out);
});

test("a fence that is not a cell stays prose, and an output with prose before it is not taken", () => {
  const b = parse("```json\n{}\n```\n\n```bash\nls\n```\n\ntext\n\n```output\nx\n```\n");
  assert.equal(b[0].type, "prose");
  assert.equal(b[1].kind, "bash");
  assert.equal(b[1].output, null);
  assert.match(b[2].text, /```output/);
});

test("aliases name the same kinds", () => {
  const b = parse("```py\n1\n```\n\n```sh\nls\n```\n\n```claude\nq\n```\n");
  assert.deepEqual(b.map((x) => x.kind), ["python", "bash", "ask"]);
});

test("the command line", () => {
  assert.deepEqual(command("web lipid standards"), { kind: "web", source: "lipid standards" });
  assert.deepEqual(command("$ ls -la"), { kind: "bash", source: "ls -la" });
  assert.deepEqual(command("py print(1)"), { kind: "python", source: "print(1)" });
  assert.deepEqual(command("https://example.org/a"), { kind: "read", source: "https://example.org/a" });
  assert.deepEqual(command("what changed in DCAT-AP 3?"), { kind: "ask", source: "what changed in DCAT-AP 3?" });
  assert.deepEqual(command("web"), { kind: "ask", source: "web" });
  assert.equal(command("   "), null);
});

test("the starter document parses into cells", () => {
  const b = parse(STARTER);
  assert.deepEqual(b.filter((x) => x.type === "cell").map((x) => x.kind), ["ask", "web", "python"]);
});

test("the record only grows, and is read newest first", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "nb-"));
  appendRecord({ at: "1", document: "a", kind: "web", source: "x", ok: true, ms: 1, output: "o" }, dir);
  appendRecord({ at: "2", document: "a", kind: "web", source: "y", ok: true, ms: 1, output: "o" }, dir);
  appendRecord({ at: "3", document: "b", kind: "web", source: "z", ok: true, ms: 1, output: "o" }, dir);
  const r = readRecord({ document: "a" }, dir);
  assert.equal(r.count, 2);
  assert.deepEqual(r.entries.map((e) => e.at), ["2", "1"]);
});

const b64 = (s) => Buffer.from(s).toString("base64url");

test("the account is read from the token's payload", () => {
  assert.equal(tokenSubject(`Bearer v1.${b64("session:acc-1:1:2:n")}.mac`), "acc-1");
  assert.equal(tokenSubject("Bearer junk"), null);
});

test("from elsewhere, only the owner may run the document", async () => {
  const local = { socket: { remoteAddress: "127.0.0.1" }, headers: {} };
  assert.deepEqual(await owner(local, {}), { ok: true, local: true });
  const remote = { socket: { remoteAddress: "203.0.113.9" }, headers: {} };
  const r = await owner(remote, {});
  assert.equal(r.ok, false);
  assert.equal(r.status, 401);
});

test("a script runs in the notebook's work folder, without the server's secrets", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "nb-"));
  process.env.NOTEBOOK_TEST_SECRET = "s3cret";
  const r = await runScript("node", "console.log(process.env.NOTEBOOK_TEST_SECRET ?? 'absent'); console.log(process.cwd().endsWith('work'))", { dir });
  assert.equal(r.code, 0);
  assert.equal(r.stdout.trim(), "absent\ntrue");
  assert.match(scriptOutput(r), /exit 0/);
  const slow = await runScript("node", "setTimeout(() => {}, 5000)", { dir, timeoutS: 1 });
  assert.equal(slow.timed_out, true);
  assert.match(scriptOutput(slow), /took too long/);
});

test("files are found by name and by content", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nb-files-"));
  fs.mkdirSync(path.join(root, "lipids"));
  fs.writeFileSync(path.join(root, "lipids", "plate-layout.csv"), "well,sample\nA1,PC 34:1 blank\n");
  fs.mkdirSync(path.join(root, "node_modules"));
  fs.writeFileSync(path.join(root, "node_modules", "blank.txt"), "PC 34:1 blank");
  const r = await searchFiles("34:1 blank", { roots: [root] });
  assert.equal(r.lines.length, 1);
  assert.equal(r.lines[0].line, 2);
  const byName = await searchFiles("lipids plate", { roots: [root] });
  assert.equal(byName.names.length, 1);
  assert.match(byName.names[0], /plate-layout\.csv$/);
});

test("a model is picked from what is configured, and a cell may ask for one", () => {
  assert.equal(pickProvider("", { ANTHROPIC_API_KEY: "k", HUGGINGFACE_API_KEY: "h" }), "claude");
  assert.equal(pickProvider("hf", { ANTHROPIC_API_KEY: "k", HUGGINGFACE_API_KEY: "h" }), "huggingface");
  assert.equal(pickProvider("", { HUGGINGFACE_API_KEY: "h" }), "huggingface");
  assert.equal(pickProvider("claude", { HUGGINGFACE_API_KEY: "h" }), null);
  assert.equal(pickProvider("", {}), null);
});

test("an ask runs Claude's tool calls and keeps the final answer, with what it looked at", async (t) => {
  const { mockAnthropic } = await import("./fixtures/mock-anthropic.mjs");
  const { askClaude } = await import("../src/lib/server/notebook-ask.js");
  const web = await import("../src/lib/server/web.js");
  const mock = await mockAnthropic();
  t.after(() => mock.server.close());
  const saved = { ...process.env };
  t.after(() => { process.env = saved; });
  Object.assign(process.env, { ANTHROPIC_API_KEY: "test-key", ANTHROPIC_BASE_URL: mock.url, NOTEBOOK_FALLBACKS: "on" });
  // No network in a test: the search engine is DuckDuckGo's HTML, served from a fixture.
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (u, o) => (String(u).includes("duckduckgo")
    ? new Response('<div class="result"><a class="result__a" href="https://nfdi-de.github.io/dcat-ap-plus/latest/">DCAT-AP+</a><a class="result__snippet">a provenance layer</a></div>', { status: 200 })
    : realFetch(u, o));
  t.after(() => { globalThis.fetch = realFetch; });

  const events = [];
  const r = await askClaude("what does DCAT-AP+ add?", { local: false, emit: (e) => events.push(e) });
  assert.match(r.output, /^DCAT-AP\+ adds provenance/);
  assert.match(r.output, /looked at: `web "DCAT-AP-PLUS"`/);
  assert.ok(events.some((e) => e.type === "step" && /web "DCAT-AP-PLUS"/.test(e.text)));

  const [first, second] = mock.requests;
  assert.equal(first.body.model, "claude-opus-5-5");
  assert.deepEqual(first.body.thinking, { type: "adaptive" });
  assert.equal(first.body.fallbacks, "default");
  assert.match(first.headers["anthropic-beta"], /server-side-fallback-2026-07-01/);
  assert.equal(first.headers["x-api-key"], "test-key");
  // Off this computer, Claude gets no file tools.
  assert.deepEqual(first.body.tools.map((x) => x.name), ["web_search", "read_page", "search_library"]);
  const result = second.body.messages.at(-1).content[0];
  assert.equal(result.type, "tool_result");
  assert.equal(result.tool_use_id, "tu1");
  assert.match(result.content, /provenance layer/);
});
