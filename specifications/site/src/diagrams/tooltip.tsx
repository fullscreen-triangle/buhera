import { useState, type MouseEvent } from "react";

// A tiny cursor-following tooltip shared by the SVG diagrams.
export function useTooltip() {
  const [state, setState] = useState<{ x: number; y: number; text: string } | null>(null);
  return {
    show: (e: MouseEvent, text: string) => setState({ x: e.clientX, y: e.clientY, text }),
    move: (e: MouseEvent) => setState((s) => (s ? { ...s, x: e.clientX, y: e.clientY } : s)),
    hide: () => setState(null),
    node: state ? (
      <div className="tooltip" style={{ left: Math.min(state.x + 14, window.innerWidth - 340), top: state.y + 14 }}>
        {state.text}
      </div>
    ) : null,
  };
}
