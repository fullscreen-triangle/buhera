/* ============================================================================
 * SandboxFrame — shared chrome for the four CKG sandbox IDEs.
 *
 * Gives every panel the same header bar plus two view controls, without any
 * panel having to reimplement them:
 *
 *   • Fullscreen  — the whole IDE fills the tab/window (a fixed overlay, inset 0,
 *                   height 100dvh) and reverts. Esc also reverts. While
 *                   fullscreen, body scroll is locked. Because each panel's 3D /
 *                   chart views size themselves off their container (via
 *                   ResizeObserver), they grow to fill the enlarged frame for
 *                   free.
 *   • Collapse editor — hide the left source column so the output/renderer takes
 *                   the full width (most useful for the 3D graph + charts), and
 *                   restore it.
 *
 * The frame owns the header and the container; the panel renders its own body
 * through a render-prop, reading { fullscreen, editorCollapsed } to drop its
 * editor column when collapsed:
 *
 *   <SandboxFrame title="SBS" subtitle="…" accent="#4ec9b0">
 *     {({ fullscreen, editorCollapsed }) => ( … body … )}
 *   </SandboxFrame>
 * ========================================================================== */

import { useState, useEffect, useCallback } from "react";
import { Maximize2, Minimize2, PanelLeftClose, PanelLeftOpen } from "lucide-react";

const BASE_HEIGHT = 560;

export default function SandboxFrame({
  title,
  subtitle,
  accent = "#4ec9b0",
  headerBg = "#101014",
  border = "#262626",
  background = "#0a0a0a",
  headerExtra = null,   // optional node rendered at the right of the header (before controls)
  children,             // render-prop: ({ fullscreen, editorCollapsed }) => body
}) {
  const [fullscreen, setFullscreen] = useState(false);
  const [editorCollapsed, setEditorCollapsed] = useState(false);

  const exit = useCallback(() => setFullscreen(false), []);

  // Esc exits fullscreen; lock body scroll while the overlay is up.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e) => { if (e.key === "Escape") exit(); };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [fullscreen, exit]);

  const containerStyle = fullscreen
    ? { position: "fixed", inset: 0, zIndex: 60, height: "100dvh", width: "100vw", background, borderRadius: 0 }
    : { position: "relative", height: BASE_HEIGHT, background };

  const ctrlBtn = {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: 24, width: 24, borderRadius: 4, color: "#9ca3af",
  };

  return (
    <div
      className="flex flex-col overflow-hidden border"
      style={{ ...containerStyle, borderColor: border, borderRadius: fullscreen ? 0 : 6 }}
    >
      {/* Header */}
      <div
        className="flex h-8 shrink-0 items-center gap-3 px-3"
        style={{ background: headerBg, borderBottom: `1px solid ${border}` }}
      >
        <span className="font-mono text-[12px] font-bold" style={{ color: accent }}>{title}</span>
        {subtitle && <span className="truncate text-[11px] text-neutral-400">{subtitle}</span>}

        <div className="ml-auto flex items-center gap-1">
          {headerExtra}
          <button
            type="button"
            onClick={() => setEditorCollapsed((c) => !c)}
            title={editorCollapsed ? "Show source column" : "Hide source column"}
            aria-label={editorCollapsed ? "Show source column" : "Hide source column"}
            className="hover:bg-white/10"
            style={ctrlBtn}
          >
            {editorCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
          </button>
          <button
            type="button"
            onClick={() => setFullscreen((f) => !f)}
            title={fullscreen ? "Exit fullscreen (Esc)" : "Fill the window"}
            aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            className="hover:bg-white/10"
            style={{ ...ctrlBtn, color: fullscreen ? accent : "#9ca3af" }}
          >
            {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex min-h-0 flex-1">
        {children({ fullscreen, editorCollapsed })}
      </div>
    </div>
  );
}
