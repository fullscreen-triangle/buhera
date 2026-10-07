/* ============================================================================
 * The small controls every surface page uses: a labelled row, a switch, a
 * slider, a choice, a button. Pages bind them to a store or to a step
 * (SurfaceActions); a control never changes the page it sits on.
 * ========================================================================== */

export function Row({ label, hint, children }) {
  return (
    <div className="grid grid-cols-[12rem_1fr] md:grid-cols-1 gap-x-6 gap-y-1 py-2 border-t border-gray-900 items-center">
      <div>
        <div className="text-gray-300 text-sm">{label}</div>
        {hint && <div className="text-gray-600 text-[11px] leading-snug">{hint}</div>}
      </div>
      <div className="text-sm">{children}</div>
    </div>
  );
}

export function Toggle({ on, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={on} onClick={() => onChange(!on)}
      className="inline-flex items-center gap-2 text-gray-300 hover:text-white">
      <span className={`w-8 h-4 rounded-full relative transition-colors ${on ? "bg-teal-600" : "bg-gray-700"}`}>
        <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-black transition-all ${on ? "left-4" : "left-0.5"}`} />
      </span>
      <span className="text-xs text-gray-500">{label ?? (on ? "on" : "off")}</span>
    </button>
  );
}

export function Slider({ value, min, max, step, onChange, format = (v) => v }) {
  return (
    <span className="inline-flex items-center gap-3">
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="w-48 accent-teal-500" />
      <span className="text-xs text-gray-400 w-12" style={{ fontVariantNumeric: "tabular-nums" }}>{format(value)}</span>
    </span>
  );
}

export function Choice({ value, options, onChange }) {
  return (
    <span className="inline-flex gap-1">
      {options.map((o) => (
        <button key={o} type="button" onClick={() => onChange(o)}
          className={`px-2 py-0.5 rounded text-xs border ${o === value ? "border-teal-600 text-teal-300" : "border-gray-800 text-gray-400 hover:border-gray-600"}`}>
          {o}
        </button>
      ))}
    </span>
  );
}

export function Button({ onClick, children, disabled }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className="px-2 py-0.5 rounded text-xs border border-gray-800 text-gray-300 hover:border-teal-600 hover:text-teal-300 disabled:opacity-40">
      {children}
    </button>
  );
}

export const pct = (v) => `${Math.round(v * 100)}%`;
export const when = (t) => new Date(t).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });

