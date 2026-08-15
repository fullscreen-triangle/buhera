/* ============================================================================
 * SbsSandboxPanel — the REAL SBS (Systems Biology Shaders) sandbox, embedded.
 *
 * This is NOT a text echo. It runs the real SBS interpreter in the browser:
 * `compileSBS(source)` (tokenize -> parse -> emit circuit + GLSL + JS + AST),
 * then `solveCircuit` (WebGL2, CPU fallback) + `extractMetrics`. The output is
 * rendered with the SAME renderer components the live `sbs-sandbox` page uses —
 * `NetworkGraph3D` (react-force-graph-3d), `ChartsPanel` (~15 D3 charts), and a
 * Compiled tab exposing the raw glsl / js / ast / circuit artifacts.
 *
 * Lifted verbatim from hegel/consequences/src/components/sbs/Sandbox.js; only
 * the full-page VS-Code chrome (activity bar, file tree, screen-height layout)
 * is dropped so it fits inside the CKG report as a bounded panel. Seeded with
 * the real KEGG hsa00190 oxidative-phosphorylation circuit, including the
 * rotenone Complex-I perturbation (NADH->CoQ factor 0.3).
 * ========================================================================== */

import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";
import {
  Eye, Code2, Trash2, BarChart3, Play,
  Terminal as TerminalIcon,
} from "lucide-react";
import {
  compileSBS, solveCircuit, extractMetrics, SCRIPTS,
} from "@sachikonye/sbs";
import ChartsPanel from "./Charts";
import SandboxFrame from "../SandboxFrame";

const theme = {
  editor: "#0f0f1a", editorFg: "#d4d4d4",
  tabActive: "#0f0f1a", tabInactive: "#1a1a2e",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#2a2a4a",
  accent: "#0e639c", accentBright: "#007acc", gutter: "#4a4a6a",
};

// The real KEGG hsa00190 oxidative-phosphorylation circuit — pulled from the
// interpreter's own script registry so it stays in lockstep with the source.
const OXPHOS_SOURCE =
  (SCRIPTS && SCRIPTS["oxphos.sbs"] && SCRIPTS["oxphos.sbs"].code) ||
  `import oxphos from "kegg/hsa00190"\ncircuit electron_transport {\n  node NADH { mu: -320.0, concentration: 0.1, compartment: "mitochondria_matrix" }\n  node CoQ  { mu: -50.0, concentration: 0.05, compartment: "inner_membrane" }\n  node O2   { mu: 815.0, concentration: 0.26, compartment: "mitochondria_matrix" }\n  edge NADH -> CoQ { conductance: 12.0 }\n  edge CoQ  -> O2  { conductance: 8.0 }\n}\nobserve electron_transport\nperturb electron_transport { edge: "NADH->CoQ", factor: 0.3 }\nnavigate from CoQ\n`;

/* ── GLB 3D viewer (three.js) — used when a script declares a glbModel ── */
function GLBViewer({ glbModel, circuit }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!glbModel?.file || !containerRef.current) return;
    let THREE, GLTFLoader, OrbitControls;
    try {
      THREE = require("three");
      GLTFLoader = require("three/examples/jsm/loaders/GLTFLoader").GLTFLoader;
      OrbitControls = require("three/examples/jsm/controls/OrbitControls").OrbitControls;
    } catch { return; }

    const el = containerRef.current;
    const w = el.clientWidth || 600;
    const h = el.clientHeight || 400;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    camera.position.set(3, 2, 4);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    scene.add(new THREE.AmbientLight(0x404040, 2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(3, 5, 3);
    scene.add(dirLight);

    const loader = new GLTFLoader();
    loader.load(glbModel.file, (gltf) => {
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      model.position.sub(center);
      model.scale.setScalar(2.5 / maxDim);
      scene.add(model);
      if (glbModel.species_map && circuit) {
        const markerGeom = new THREE.SphereGeometry(0.08, 16, 16);
        for (const [, info] of Object.entries(glbModel.species_map)) {
          const mat = new THREE.MeshPhongMaterial({ color: info.color || "#4ec9b0", emissive: info.color || "#4ec9b0", emissiveIntensity: 0.3 });
          const marker = new THREE.Mesh(markerGeom, mat);
          marker.position.set(...(info.position || [0, 0, 0]));
          scene.add(marker);
        }
      }
    }, undefined, (err) => { console.warn("GLB load error:", err); });

    let animId;
    const animate = () => { animId = requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      controls.dispose();
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, [glbModel, circuit]);

  return <div ref={containerRef} className="h-full w-full" style={{ background: "#0a0a14" }} />;
}

/* ── 3D Network Graph (react-force-graph-3d) ── */
function NetworkGraph3D({ circuit, metrics }) {
  const containerRef = useRef(null);
  const fgRef = useRef(null);
  const [ForceGraph, setForceGraph] = useState(null);
  const [graphLoadError, setGraphLoadError] = useState(false);
  const [dims, setDims] = useState({ width: 600, height: 400 });

  useEffect(() => {
    import("react-force-graph-3d")
      .then(mod => setForceGraph(() => mod.default || mod))
      .catch(() => setGraphLoadError(true));
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setDims({ width, height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!circuit) return { nodes: [], links: [] };
    const Se = metrics?.Se || [];
    const muMin = Math.min(...circuit.nodes.map(n => n.mu));
    const muMax = Math.max(...circuit.nodes.map(n => n.mu));
    const muRange = muMax - muMin || 1;
    const nodes = circuit.nodes.map((n, i) => ({
      id: i, name: n.name, mu: n.mu, concentration: n.concentration,
      compartment: n.compartment, se: Se[i] || 0, muNorm: (n.mu - muMin) / muRange,
    }));
    const links = circuit.edges.map(e => ({
      source: e.src, target: e.dst, conductance: e.conductance, name: e.name,
    }));
    return { nodes, links };
  }, [circuit, metrics]);

  const maxG = useMemo(() => {
    if (!circuit) return 1;
    return Math.max(1, ...circuit.edges.map(e => e.conductance));
  }, [circuit]);

  const nodeColor = useCallback((node) => {
    const t = node.muNorm;
    const r = Math.round(255 * Math.max(0, Math.min(1, 0.267 + 2.82 * t * t - 5.77 * t ** 3 + 2.68 * t ** 4)));
    const g = Math.round(255 * Math.max(0, Math.min(1, 0.005 + 1.4 * t - 2.8 * t * t + 4.4 * t ** 3 - 2 * t ** 4)));
    const b = Math.round(255 * Math.max(0, Math.min(1, 0.329 + 1.07 * t - 0.73 * t * t + 0.33 * t ** 3)));
    return `rgb(${r},${g},${b})`;
  }, []);

  const nodeLabel = useCallback((node) => {
    return `<div style="background:rgba(10,10,20,0.9);padding:6px 10px;border-radius:4px;border:1px solid #2a2a4a;font-family:monospace;font-size:11px;">
      <div style="color:#4ec9b0;font-weight:bold;margin-bottom:2px;">${node.name}</div>
      <div style="color:#888;">μ = ${node.mu.toFixed(1)} kJ/mol</div>
      <div style="color:#888;">conc = ${node.concentration}</div>
      <div style="color:#888;">compartment: ${node.compartment}</div>
      <div style="color:#888;">Se = ${node.se.toFixed(3)}</div>
    </div>`;
  }, []);

  const nodeThreeObject = useCallback((node) => {
    const THREE = require("three");
    const group = new THREE.Group();
    const geom = new THREE.SphereGeometry(4, 16, 16);
    const color = nodeColor(node);
    const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.3, transparent: true, opacity: 0.9 });
    group.add(new THREE.Mesh(geom, mat));
    const canvas = document.createElement("canvas");
    canvas.width = 128; canvas.height = 32;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 18px monospace";
    ctx.textAlign = "center";
    ctx.fillText(node.name.length > 10 ? node.name.slice(0, 9) + "…" : node.name, 64, 22);
    const tex = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(24, 6, 1);
    sprite.position.set(0, 7, 0);
    group.add(sprite);
    return group;
  }, [nodeColor]);

  const linkWidth = useCallback((link) => Math.max(0.5, 3 * link.conductance / maxG), [maxG]);
  const linkColor = useCallback(() => "rgba(74,158,255,0.3)", []);

  if (!ForceGraph) {
    return (
      <div ref={containerRef} className="flex h-full w-full items-center justify-center" style={{ background: "#0a0a14" }}>
        <span style={{ color: graphLoadError ? "#ef4444" : "#5a5a5a" }}>
          {graphLoadError ? "Failed to load 3D graph" : "Loading 3D graph…"}
        </span>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full w-full" style={{ background: "#0a0a14" }}>
      <ForceGraph
        ref={fgRef}
        width={dims.width}
        height={dims.height}
        graphData={graphData}
        backgroundColor="#0a0a14"
        nodeThreeObject={nodeThreeObject}
        nodeLabel={nodeLabel}
        linkWidth={linkWidth}
        linkColor={linkColor}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={0.85}
        linkDirectionalArrowColor={() => "rgba(74,158,255,0.6)"}
        linkCurvature={0.1}
        linkOpacity={0.6}
        enableNodeDrag={true}
        enableNavigationControls={true}
        showNavInfo={false}
      />
    </div>
  );
}

/* ── Preview panel — 3D network graph + optional GLB ── */
function PreviewPanel({ circuit, metrics, glbModel }) {
  const [showGLB, setShowGLB] = useState(false);
  useEffect(() => { setShowGLB(!!glbModel?.file); }, [glbModel]);

  if (!circuit) {
    return <div className="flex h-full items-center justify-center text-sm" style={{ color: "#5a5a5a" }}>Run a script to see the circuit</div>;
  }

  return (
    <div className="relative h-full w-full" style={{ background: "#0a0a14" }}>
      {showGLB && glbModel ? (
        <GLBViewer glbModel={glbModel} circuit={circuit} />
      ) : (
        <NetworkGraph3D circuit={circuit} metrics={metrics} />
      )}
      <div className="absolute left-3 top-3 flex items-center gap-2" style={{ pointerEvents: "none" }}>
        <span className="rounded px-2 py-1 text-[11px] font-mono" style={{ background: "rgba(15,15,26,0.85)", color: "#4ec9b0" }}>
          {circuit.numNodes} nodes, {circuit.numEdges} edges
          {showGLB ? " — GLB model" : " — 3D force graph"}
        </span>
        {glbModel?.file && (
          <button
            onClick={() => setShowGLB(s => !s)}
            className="rounded px-2 py-1 text-[11px] font-mono"
            style={{ background: "rgba(15,15,26,0.85)", color: showGLB ? "#dcdcaa" : "#4ec9b0", border: "1px solid #2a2a4a", pointerEvents: "auto" }}
          >
            {showGLB ? "Show Graph" : "Show 3D"}
          </button>
        )}
      </div>
      {circuit.compartments && circuit.compartments.length > 1 && (
        <div className="absolute bottom-3 left-3 flex gap-2" style={{ pointerEvents: "none" }}>
          {circuit.compartments.map(c => (
            <span key={c} className="rounded-full px-2 py-0.5 text-[10px]" style={{ background: "#1a1a3e", color: "#888" }}>{c}</span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Output column — the Preview / Charts / Console / Compiled tabs ── */
function OutputColumn({ circuit, metrics, compiled, logs, errors, imports, glbModel, onRun, onClear }) {
  const [tab, setTab] = useState("preview");
  const [compiledSub, setCompiledSub] = useState("glsl");
  const tabs = [
    { id: "preview", label: "Preview", Icon: Eye },
    { id: "charts", label: "Charts", Icon: BarChart3 },
    { id: "console", label: "Console", Icon: TerminalIcon },
    { id: "compiled", label: "Compiled", Icon: Code2 },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };
  const compiledText = {
    glsl: compiled.glsl || "// No GLSL output",
    js: compiled.js || "// No JS output",
    ast: compiled.ast || "// No AST",
    circuit: compiled.circuit || "// No circuit",
  };

  return (
    <div className="flex min-w-0 flex-1 flex-col" style={{ background: theme.editor, borderLeft: `1px solid ${theme.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: theme.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button key={id} onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? theme.tabFgActive : theme.tabFg, background: active ? theme.tabActive : "transparent" }}>
                <Icon size={13} /> {label}
                {id === "console" && logs.length > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]" style={{ background: theme.accent, color: "#fff" }}>{logs.length}</span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {tab === "console" && (
            <button onClick={onClear} title="Clear console" className="flex h-6 w-6 items-center justify-center rounded" style={{ color: theme.tabFg }}><Trash2 size={14} /></button>
          )}
          <button onClick={onRun} title="Run (Ctrl+Enter)" className="flex h-6 items-center gap-1 rounded px-2 text-[12px]" style={{ background: "#0e639c", color: "#fff" }}>
            <Play size={12} /> Run
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === "preview" && <PreviewPanel circuit={circuit} metrics={metrics} glbModel={glbModel} />}
        {tab === "charts" && <ChartsPanel metrics={metrics} circuit={circuit} imports={imports} />}
        {tab === "console" && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-relaxed">
            {errors.length > 0 && errors.map((e, i) => (
              <div key={`err${i}`} className="border-b px-1 py-1" style={{ color: "#f48771", borderColor: "#2a2a2a" }}>
                <span className="mr-2 opacity-50">error</span>{e.message}{e.line ? ` (line ${e.line})` : ""}
              </div>
            ))}
            {logs.length === 0 && errors.length === 0 ? (
              <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>Console output appears here.</div>
            ) : logs.map((l, i) => (
              <div key={i} className="border-b px-1 py-1" style={{ color: levelColor[l.level] || "#d4d4d4", borderColor: "#2a2a2a" }}>
                <span className="mr-2 opacity-50">{l.level}</span>{l.message}
              </div>
            ))}
          </div>
        )}
        {tab === "compiled" && (
          <div className="h-full overflow-auto">
            <div className="flex border-b" style={{ borderColor: theme.border }}>
              {["glsl", "js", "ast", "circuit"].map(sub => (
                <button key={sub} onClick={() => setCompiledSub(sub)}
                  className="px-3 py-1.5 text-[11px] uppercase tracking-wider"
                  style={{ color: compiledSub === sub ? theme.tabFgActive : theme.tabFg }}>
                  {sub}
                </button>
              ))}
            </div>
            <pre className="p-3 font-mono text-[12px] leading-[1.5] whitespace-pre-wrap" style={{ color: theme.editorFg }}>
              {compiledText[compiledSub]}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Compact editor ── */
function Editor({ value, onChange }) {
  const gutterRef = useRef(null);
  const lines = value.split("\n");
  const syncScroll = (e) => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  const handleKeyDown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const { selectionStart, selectionEnd } = e.target;
      const next = value.slice(0, selectionStart) + "  " + value.slice(selectionEnd);
      onChange(next);
      requestAnimationFrame(() => { e.target.selectionStart = e.target.selectionEnd = selectionStart + 2; });
    }
  };
  return (
    <div className="flex min-h-0 flex-1" style={{ background: theme.editor }}>
      <div ref={gutterRef} className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]" style={{ color: theme.gutter, minWidth: 44, paddingRight: 12 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={(e) => onChange(e.target.value)} onScroll={syncScroll}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: theme.editorFg, tabSize: 2, caretColor: "#4ec9b0" }}
      />
    </div>
  );
}

/* ── The panel ── */
export default function SbsSandboxPanel() {
  const [source, setSource] = useState(OXPHOS_SOURCE);
  const [circuit, setCircuit] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [compiled, setCompiled] = useState({ glsl: null, js: null, ast: null, circuit: null });
  const [logs, setLogs] = useState([]);
  const [errors, setErrors] = useState([]);
  const [imports, setImports] = useState({});
  const [glbModel, setGlbModel] = useState(null);

  const run = useCallback((code) => {
    const src = typeof code === "string" ? code : source;
    if (!src) return;

    const t0 = performance.now();
    const result = compileSBS(src);
    const compileTime = performance.now() - t0;

    const newLogs = [];
    const newErrors = [];

    if (!result.success) {
      setErrors(result.errors || []);
      setLogs([]);
      setCircuit(null); setMetrics(null); setImports({}); setGlbModel(null);
      setCompiled({ glsl: null, js: null, ast: null, circuit: null });
      return;
    }

    if (result.imports && Object.keys(result.imports).length > 0) {
      setImports(result.imports);
      for (const [name, data] of Object.entries(result.imports)) {
        newLogs.push({ level: "info", message: `import ${name} from ${data.source || "registry"}: ${data.name || data.id || "resolved"}` });
      }
    } else setImports({});

    setGlbModel(result.glbModel || null);
    newLogs.push({ level: "info", message: `Compiled in ${compileTime.toFixed(1)}ms` });

    setCompiled({
      glsl: result.glsl || null,
      js: result.js || null,
      ast: result.ast ? JSON.stringify(result.ast, null, 2) : null,
      circuit: result.circuit ? JSON.stringify(result.circuit, null, 2) : null,
    });

    if (result.circuit) {
      newLogs.push({ level: "log", message: `Circuit: ${result.circuit.numNodes} nodes, ${result.circuit.numEdges} edges` });
      newLogs.push({ level: "log", message: `Compartments: ${result.circuit.compartments?.join(", ")}` });

      const solverPerts = [];
      if (result.perturbations?.length > 0) {
        for (const p of result.perturbations) {
          if (p.edge) {
            const edgeIdx = result.circuit.edges.findIndex(e => e.name === p.edge);
            if (edgeIdx >= 0) solverPerts.push({ idx: edgeIdx, factor: p.factor });
          } else {
            for (let i = 0; i < result.circuit.numEdges; i++) solverPerts.push({ idx: i, factor: p.factor });
          }
        }
      }

      try {
        const shaderResult = solveCircuit(result.circuit, solverPerts.length > 0 ? solverPerts : null);
        const healthyResult = solveCircuit(result.circuit, null);
        const m = extractMetrics(shaderResult, result.circuit, healthyResult, solverPerts.length > 0 ? solverPerts : null);
        setMetrics({
          ...m,
          numNodes: result.circuit.numNodes,
          numEdges: result.circuit.numEdges,
          backend: shaderResult.backend || "cpu",
          renderTimeMs: shaderResult.renderTimeMs,
        });
        setCircuit(result.circuit);
        newLogs.push({ level: "log", message: `R (coherence): ${m.R?.toFixed(4)}` });
        newLogs.push({ level: "log", message: `V (visibility): ${m.V?.toFixed(4)}` });
        newLogs.push({ level: "info", message: `Solved on ${shaderResult.backend} in ${shaderResult.renderTimeMs?.toFixed(1)}ms` });
        if (result.perturbations?.length > 0) newLogs.push({ level: "warn", message: `${result.perturbations.length} perturbation(s) active` });
        if (result.observations?.length > 0) newLogs.push({ level: "info", message: `${result.observations.length} observation(s)` });
      } catch (e) {
        newErrors.push({ message: `Solver error: ${e.message}`, line: 0 });
        setMetrics(null);
        setCircuit(result.circuit);
      }
    } else {
      setCircuit(null); setMetrics(null);
    }

    setErrors(newErrors);
    setLogs(newLogs);
  }, [source]);

  // Auto-run the seeded oxphos circuit on mount.
  useEffect(() => { run(OXPHOS_SOURCE); /* eslint-disable-next-line */ }, []);

  return (
    <SandboxFrame
      title="SBS"
      subtitle="Systems Biology Shaders — oxidative phosphorylation (KEGG hsa00190)"
      accent="#4ec9b0"
      headerBg="#16213e"
      border={theme.border}
      background={theme.editor}
    >
      {({ editorCollapsed }) => (
        <>
          {!editorCollapsed && (
            <div className="flex min-w-0 flex-col" style={{ width: "44%" }}>
              <Editor value={source} onChange={setSource} />
            </div>
          )}
          <OutputColumn
            circuit={circuit} metrics={metrics} compiled={compiled}
            logs={logs} errors={errors} imports={imports} glbModel={glbModel}
            onRun={() => run(source)} onClear={() => setLogs([])}
          />
        </>
      )}
    </SandboxFrame>
  );
}
