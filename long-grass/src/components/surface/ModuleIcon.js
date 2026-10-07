/* ============================================================================
 * ModuleIcon — one mark per module, wherever a module is named.
 *
 * Known modules get a pictogram chosen for what they do (lucide, 1.5px
 * strokes to match the surface's hairlines). A module this table has never
 * heard of still gets a mark: a glyph generated from its id — three arms on
 * the ternary axes, lengths from a hash of the name — so a new module is
 * never blank and always looks the same.
 * ========================================================================== */

import {
  Wifi, SprayCan, Triangle, Database, Server, Boxes, Cpu, Network, Waypoints, CircuitBoard, Cable,
  HardDrive, SlidersHorizontal, RotateCw, CloudDownload, ChartScatter, MemoryStick, Repeat,
  FlaskConical, BrainCircuit, Terminal, Route, Sparkles, Languages, Crosshair, Shapes, Microscope,
  NotebookPen, Braces, Hammer, Workflow, Pill, Bot, Layers, Timer, Waves, Radar, Wind,
  MonitorSmartphone, Microchip, FlaskRound, TramFront, ALargeSmall, Monitor, Code, Printer, UserCog,
  FolderKanban, LibraryBig, ListChecks, FileText, Mail, ServerCog, ChartGantt, Globe, DraftingCompass,
} from "lucide-react";

export const ICONS = {
  // top — where the session is
  devices: MonitorSmartphone,
  machine: Microchip,
  network: Wifi,
  web: Globe,
  mail: Mail,
  experiments: FlaskRound,
  lattice: ServerCog,
  runtime: TramFront,
  // right — how the screen is used
  preferences: ALargeSmall,
  screen: Monitor,
  code: Code,
  peripherals: Printer,
  // bottom — who the work is for
  model: UserCog,
  projects: FolderKanban,
  rag: LibraryBig,
  planning: ChartGantt,
  spec: DraftingCompass,
  plans: ListChecks,
  reports: FileText,
  // the federation (left column)
  spraypaint: SprayCan,
  triangle: Triangle,
  hfq: Database,
  gateway: Server,
  catalysts: Boxes,
  compute: Cpu,
  srn: Network,
  pylon: Waypoints,
  "sbs-core": Cable,
  disk: HardDrive,
  config: SlidersHorizontal,
  restart: RotateCw,
  update: CloudDownload,
  vis: ChartScatter,
  vahera: MemoryStick,
  echo: Repeat,
  lavoisier: FlaskConical,
  purpose: BrainCircuit,
  "purpose-cli": Terminal,
  "purpose-carry": Route,
  zangalewa: Sparkles,
  "zangalewa-dsl": Languages,
  graffiti: Crosshair,
  shapeshifter: Shapes,
  scope: Microscope,
  desk: NotebookPen,
  "dsl-writer": Braces,
  smith: Hammer,
  ckg: Workflow,
  cytochrome: Pill,
  interceptor: Bot,
  ladder: Layers,
  sbs: CircuitBoard,
  tempus: Timer,
  ndombolo: Waves,
  tracker: Radar,
  windtunnel: Wind,
};

// FNV-1a: a stable hash of the id, so a generated glyph never changes.
function hash(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Three arm lengths in [0.35, 1] for an id — its generated address. */
export function glyphArms(id) {
  const h = hash(String(id));
  return [0, 1, 2].map((i) => 0.35 + (((h >>> (i * 10)) & 0x3ff) / 0x3ff) * 0.65);
}

function AddressGlyph({ id, size }) {
  const arms = glyphArms(id);
  const c = 12;
  const pts = arms.map((len, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / 3; // ternary axes: up, lower-right, lower-left
    return [c + Math.cos(a) * len * 9, c + Math.sin(a) * len * 9];
  });
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points={pts.map((p) => p.join(",")).join(" ")} strokeOpacity={0.45} />
      {pts.map(([x, y], i) => <line key={i} x1={c} y1={c} x2={x} y2={y} />)}
      <circle cx={c} cy={c} r={1.6} fill="currentColor" stroke="none" />
    </svg>
  );
}

export default function ModuleIcon({ id, size = 16, className = "" }) {
  const Icon = ICONS[id];
  return (
    <span className={`inline-flex shrink-0 ${className}`} aria-hidden="true">
      {Icon ? <Icon size={size} strokeWidth={1.5} /> : <AddressGlyph id={id} size={size} />}
    </span>
  );
}
