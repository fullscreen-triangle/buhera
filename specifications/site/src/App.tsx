import { useEffect, useState, type ReactNode } from "react";
import { architecture, catalogue, LAYER_ORDER, shortSha, specs, vendor, type Binding } from "./data";
import { Markdown } from "./Markdown";
import { DispatchPipeline } from "./diagrams/DispatchPipeline";
import { FederationMap } from "./diagrams/FederationMap";
import { LanguageFlow } from "./diagrams/LanguageFlow";
import { SourcingGraph } from "./diagrams/SourcingGraph";

// Hash routes: #/  #/doc/<slug>  #/module/<id>  #/catalogue  #/sourcing
function useRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => {
      setRoute(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

const go = (path: string) => (window.location.hash = `#/${path}`);

export function BindingBadge({ b, host }: { b: Binding; host?: string }) {
  return (
    <span className={`badge b-${b}`} title={host ? `${host}: ${b}` : b}>
      {host ? `${host} ` : ""}
      {b}
    </span>
  );
}

function Legend() {
  return (
    <div className="legend">
      {(["native", "remote", "bridge", "none"] as const).map((b) => (
        <span key={b}>
          <i style={{ background: `var(--${b})` }} />
          {b}
        </span>
      ))}
      <span>
        <i style={{ background: "var(--rust)" }} />
        Rust engine
      </span>
      <span>
        <i style={{ background: "var(--ts)" }} />
        TS/JS engine
      </span>
    </div>
  );
}

function Card({ title, sub, children }: { title: string; sub?: string; children: ReactNode }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}

function Home() {
  const verified = vendor.entries.filter((e) => e.mode !== "build").length;
  return (
    <>
      <div className="hero">
        <h1>The Buhera Federation</h1>
        <p>
          How Buhera's modules and languages are integrated. Every engine lives in its own repository and is vendored
          byte-exact at a recorded commit. It is wrapped, never reimplemented, behind one module contract, and dispatched
          through one registry with the same semantics in Rust and TypeScript. Every diagram here is drawn from{" "}
          <code>catalogue.json</code> and <code>vendor.json</code>, the files the conformance tests check.
        </p>
      </div>
      <div className="stats">
        <div className="stat">
          <b>{catalogue.modules.length}</b>
          <span>modules in the library federation</span>
        </div>
        <div className="stat">
          <b>{catalogue.dsls.length}</b>
          <span>languages with real validators</span>
        </div>
        <div className="stat">
          <b>{new Set(catalogue.modules.flatMap((m) => m.upstream.map((u) => u.repo))).size}</b>
          <span>upstream repositories</span>
        </div>
        <div className="stat">
          <b>{vendor.entries.length}</b>
          <span>vendored copies ({verified} byte-verified)</span>
        </div>
        <div className="stat">
          <b>2</b>
          <span>host languages, one contract</span>
        </div>
      </div>
      <Card title="Federation map" sub="Upstream repository → module → how each host reaches it. Hover a module to isolate its paths; click to open its specification.">
        <Legend />
        <FederationMap onOpen={(id) => go(`module/${id}`)} />
      </Card>
      <Card title="One act, step by step" sub="Registry::dispatch, identical in buhera-registry (Rust) and @buhera/registry (TypeScript). Specification 03.">
        <DispatchPipeline />
      </Card>
      <Card title="Modules">
        <ModuleGrid />
      </Card>
    </>
  );
}

function ModuleGrid() {
  return (
    <div className="module-grid">
      {LAYER_ORDER.flatMap((layer) => catalogue.modules.filter((m) => m.layer === layer)).map((m) => (
        <a key={m.id} className="module-card" href={`#/module/${m.id}`}>
          <h3>{m.name}</h3>
          <div className="id">
            {m.id} · {m.layer}
            {m.dsl ? ` · ${m.dsl}` : ""}
          </div>
          <p>{m.summary}</p>
          <div className="row">
            <BindingBadge b={m.bindings.rust} host="rust" />
            <BindingBadge b={m.bindings.ts} host="ts" />
          </div>
        </a>
      ))}
    </div>
  );
}

function DocPage({ slug }: { slug: string }) {
  const doc = architecture.find((d) => d.slug === slug);
  if (!doc) return <p>Unknown document.</p>;
  const extra =
    slug.startsWith("03") ? (
      <Card title="Interactive: the dispatch pipeline">
        <DispatchPipeline />
      </Card>
    ) : slug.startsWith("04") ? (
      <Card title="The registered languages" sub="Drawn from catalogue.json · dsls">
        <LanguageFlow onOpen={(id) => go(`module/${id}`)} />
      </Card>
    ) : slug.startsWith("05") || slug.startsWith("01") ? (
      <Card title="The catalogue, drawn">
        <Legend />
        <FederationMap onOpen={(id) => go(`module/${id}`)} />
      </Card>
    ) : slug.startsWith("06") ? (
      <Card title="Every vendored copy" sub="Drawn from vendor.json. Hover an entry for its note.">
        <SourcingGraph />
      </Card>
    ) : null;
  return (
    <>
      <Markdown source={doc.body} />
      {extra}
    </>
  );
}

function ModulePage({ id }: { id: string }) {
  const m = catalogue.modules.find((x) => x.id === id);
  const spec = specs[id];
  if (!m || !spec) return <p>Unknown module.</p>;
  const langs = catalogue.dsls.filter((d) => d.module_id === id);
  return (
    <>
      <div className="module-head">
        <div className="kv">
          <span>Registry id</span>
          <b>{m.id}</b>
        </div>
        <div className="kv">
          <span>Layer</span>
          <b>{m.layer}</b>
        </div>
        <div className="kv">
          <span>Rust host</span>
          <BindingBadge b={m.bindings.rust} />
        </div>
        <div className="kv">
          <span>TypeScript host</span>
          <BindingBadge b={m.bindings.ts} />
        </div>
        <div className="kv">
          <span>Language</span>
          <b>{langs.map((l) => `${l.label} (${l.extension})`).join(", ") || "—"}</b>
        </div>
        <div className="kv">
          <span>Residue counts</span>
          <b style={{ fontWeight: 500 }}>{m.residue}</b>
        </div>
        {m.upstream.map((u) => (
          <div className="kv" key={u.path}>
            <span>Upstream</span>
            <b>
              {u.repo}@{shortSha(u.commit)}
            </b>
            <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink-3)" }}>{u.path}</div>
          </div>
        ))}
        <div className="kv">
          <span>Output kinds</span>
          <b style={{ fontFamily: "var(--mono)", fontWeight: 400, fontSize: 12 }}>{m.output_kinds.join(" · ")}</b>
        </div>
      </div>
      <Markdown source={spec.body} />
    </>
  );
}

function CataloguePage() {
  return (
    <>
      <div className="hero">
        <h1>Catalogue</h1>
        <p>
          <code>specifications/registry/catalogue.json</code> ({catalogue.schema}). Both registry libraries are
          conformance-tested against it (rules C1–C5, specification 05).
        </p>
      </div>
      <Card title="Bindings" sub="How each host reaches each engine.">
        <table className="grid">
          <thead>
            <tr>
              <th>Module</th>
              <th>Layer</th>
              <th>Language</th>
              <th>Rust</th>
              <th>TypeScript</th>
              <th>Side effects</th>
            </tr>
          </thead>
          <tbody>
            {catalogue.modules.map((m) => (
              <tr key={m.id}>
                <td>
                  <a href={`#/module/${m.id}`}>{m.id}</a>
                </td>
                <td>{m.layer}</td>
                <td>{m.dsl ?? "—"}</td>
                <td>
                  <BindingBadge b={m.bindings.rust} />
                </td>
                <td>
                  <BindingBadge b={m.bindings.ts} />
                </td>
                <td style={{ color: "var(--ink-2)", fontSize: 12.5 }}>{m.side_effects.join("; ") || "none"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card title="Languages" sub="The DSL registry: real validator, executing module, grounding pack (specification 04).">
        <LanguageFlow onOpen={(id) => go(`module/${id}`)} />
      </Card>
    </>
  );
}

function SourcingPage() {
  return (
    <>
      <div className="hero">
        <h1>Sourcing</h1>
        <p>
          Every engine enters by <code>scripts/vendor-sync.mjs</code>: copied from upstream's committed content at a
          recorded commit, and byte-checked by <code>--check</code> (specification 06).
        </p>
      </div>
      <Card title="Upstream → vendored copy → engine store">
        <SourcingGraph />
      </Card>
      <Card title="vendor.json">
        <table className="grid">
          <thead>
            <tr>
              <th>Entry</th>
              <th>Upstream path</th>
              <th>Vendored at</th>
              <th>Mode</th>
              <th>Commit</th>
            </tr>
          </thead>
          <tbody>
            {vendor.entries.map((e) => (
              <tr key={e.id}>
                <td>{e.id}</td>
                <td style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
                  {e.repo.split("/")[1]}:{e.from}
                </td>
                <td style={{ fontFamily: "var(--mono)", fontSize: 12 }}>{e.to}</td>
                <td>
                  {e.mode}
                  {e.local?.length ? " + local" : ""}
                </td>
                <td style={{ fontFamily: "var(--mono)", fontSize: 12 }}>{shortSha(e.commit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}

export function App() {
  const route = useRoute();
  const [theme, setTheme] = useState<string | null>(() => {
    try {
      return localStorage.getItem("buhera-spec-theme");
    } catch {
      return null;
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem("buhera-spec-theme", theme);
      else localStorage.removeItem("buhera-spec-theme");
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  const [page, arg] = route;
  const active = (p: string) => (route.join("/") === p ? "nav-link active" : "nav-link");
  let body: ReactNode;
  if (page === "doc" && arg) body = <DocPage slug={arg} />;
  else if (page === "module" && arg) body = <ModulePage id={arg} />;
  else if (page === "catalogue") body = <CataloguePage />;
  else if (page === "sourcing") body = <SourcingPage />;
  else body = <Home />;

  return (
    <div className="shell">
      <nav className="sidebar">
        <a className="brand" href="#/">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
            <circle cx="16" cy="16" r="13" fill="none" stroke="var(--accent)" strokeWidth="3" />
            <circle cx="16" cy="16" r="4" fill="var(--accent)" />
          </svg>
          <span>
            Buhera Federation
            <small>specifications</small>
          </span>
        </a>
        <div className="nav-group">
          <h4>Explore</h4>
          <a className={active("")} href="#/">
            Overview
          </a>
          <a className={active("catalogue")} href="#/catalogue">
            Catalogue
          </a>
          <a className={active("sourcing")} href="#/sourcing">
            Sourcing
          </a>
        </div>
        <div className="nav-group">
          <h4>Architecture</h4>
          {architecture.map((d) => (
            <a key={d.slug} className={active(`doc/${d.slug}`)} href={`#/doc/${d.slug}`}>
              <span className="num">{d.number}</span>
              {d.title.replace(/^The /, "")}
            </a>
          ))}
        </div>
        {LAYER_ORDER.map((layer) => (
          <div className="nav-group" key={layer}>
            <h4>{layer}</h4>
            {catalogue.modules
              .filter((m) => m.layer === layer)
              .map((m) => (
                <a key={m.id} className={active(`module/${m.id}`)} href={`#/module/${m.id}`}>
                  {m.id}
                </a>
              ))}
          </div>
        ))}
        <button className="theme-toggle" onClick={() => setTheme((t) => (t === "dark" ? "light" : t === "light" ? null : "dark"))}>
          Theme: {theme ?? "system"}
        </button>
      </nav>
      <main className="main">{body}</main>
    </div>
  );
}
