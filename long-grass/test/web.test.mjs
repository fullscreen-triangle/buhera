// Tests for reading the web: the address guard, HTML to Markdown with
// anchors a citation can reach, where a page is kept, and reading a search
// engine's results.
import test from "node:test";
import assert from "node:assert/strict";




import { htmlToPage, isPrivateAddress, libraryFile, parseDuckDuckGo, textToPage, fetchGuarded } from "../src/lib/server/web.js";
import { workVerb } from "../src/lib/surface/verbs.js";

test("private and loopback addresses are recognised", () => {
  for (const ip of ["127.0.0.1", "10.1.2.3", "172.20.0.1", "192.168.1.5", "169.254.1.1", "100.77.3.78", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "0.0.0.0"]) {
    assert.equal(isPrivateAddress(ip), true, ip);
  }
  for (const ip of ["185.199.108.153", "8.8.8.8", "2606:4700::1111"]) assert.equal(isPrivateAddress(ip), false, ip);
});

test("a remote request cannot make the server fetch its own network", async () => {
  await assert.rejects(fetchGuarded("http://127.0.0.1:8090/api/experiments", { local: false }), /private address/);
  await assert.rejects(fetchGuarded("file:///etc/passwd", { local: true }), /only http and https/);
});

test("HTML becomes Markdown: navigation dropped, headings anchored, links absolute", () => {
  const html = `<html><head><title>Spec</title></head><body><nav>menu</nav><main>
    <section id="intro"><h2>Introduction</h2><p>See <a href="/other">the other page</a> and <a href="#Dataset">Dataset</a>.</p>
      <section><h3>Context</h3><p>${"words ".repeat(60)}</p></section></section>
    <section id="Dataset"><h3>Dataset</h3><table><thead><tr><th>Property</th><th>Card</th></tr></thead><tbody><tr><td>title</td><td>1..n</td></tr></tbody></table></section>
    <script>evil()</script></main></body></html>`;
  const p = htmlToPage(html, "https://ex.org/spec/");
  assert.equal(p.title, "Spec");
  assert.deepEqual(p.outline.map((o) => [o.level, o.text, o.anchor]), [[2, "Introduction", "intro"], [3, "Context", "context"], [3, "Dataset", "Dataset"]]);
  assert.match(p.markdown, /## Introduction \{#intro\}/);
  assert.match(p.markdown, /\[the other page\]\(https:\/\/ex\.org\/other\)/);
  assert.match(p.markdown, /\| title \| 1\.\.n \|/);
  assert.doesNotMatch(p.markdown, /menu|evil/);
  assert.ok(p.links.includes("https://ex.org/other"));
});

test("YAML is kept as a code block, Markdown as itself", () => {
  assert.match(textToPage("classes:\n  A: {}", "https://ex.org/s/schema.yaml", "text/yaml").markdown, /^```yaml\nclasses:/);
  const md = textToPage("# Title\n\n## Part", "https://ex.org/README.md", "text/markdown");
  assert.deepEqual(md.outline.map((o) => o.text), ["Title", "Part"]);
});

test("where a page is kept", () => {
  assert.equal(libraryFile("https://semiceu.github.io/DCAT-AP/releases/3.0.1/"), "semiceu.github.io/DCAT-AP/releases/3.0.1/index.md");
  assert.equal(libraryFile("https://nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml"), "nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml.md");
  assert.equal(libraryFile("https://ex.org/a/b.html?x=1"), "ex.org/a/b.md");
});

test("a search engine's results", () => {
  const html = `<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnfdi-de.github.io%2Fdcat-ap-plus%2Flatest%2F&rut=x">DCAT-AP+</a><a class="result__snippet">A <b>LinkML</b>-based   extension.</a></div>
    <div class="result"><a class="result__a" href="https://github.com/nfdi-de/dcat-ap-plus">GitHub</a></div>
    <div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">an ad</a></div>`;
  assert.deepEqual(parseDuckDuckGo(html), [
    { title: "DCAT-AP+", url: "https://nfdi-de.github.io/dcat-ap-plus/latest/", snippet: "A LinkML-based extension." },
    { title: "GitHub", url: "https://github.com/nfdi-de/dcat-ap-plus", snippet: "" },
  ]);
});

test("the reading verbs", () => {
  assert.deepEqual(workVerb("read https://semiceu.github.io/DCAT-AP/releases/3.0.1/").instruction, { kind: "read", url: "https://semiceu.github.io/DCAT-AP/releases/3.0.1/" });
  assert.deepEqual(workVerb("read site https://nfdi-de.github.io/dcat-ap-plus/latest/ 40").instruction, { kind: "site", url: "https://nfdi-de.github.io/dcat-ap-plus/latest/", limit: 40 });
  assert.deepEqual(workVerb("web DCAT-AP-PLUS LinkML").instruction, { kind: "search", query: "DCAT-AP-PLUS LinkML" });
  const d = workVerb("diagram https://a.org/x around Distribution depth 2 all").instruction;
  assert.deepEqual([d.view, d.focus, d.depth, d.attributes], ["classes", "Distribution", 2, "all"]);
  assert.deepEqual(workVerb("workflow https://b.org/p.yaml against https://a.org/x").instruction, { kind: "diagram", view: "flow", url: "https://b.org/p.yaml", against: "https://a.org/x" });
  assert.deepEqual(workVerb("compare https://a.org/x with https://b.org/p.yaml").instruction, { kind: "compare", a: "https://a.org/x", b: "https://b.org/p.yaml" });
  assert.equal(workVerb("flowchart LR\n  A --> B").instruction.kind, "mermaid", "Mermaid is the one verb of several lines");
  assert.equal(workVerb("draw the lab workflow").instruction.request, "the lab workflow");
  assert.equal(workVerb("read the paper"), null, "without an address, reading is the player's");
});


