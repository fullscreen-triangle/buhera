// @buhera/spraypaint/react — React + D3 chart components for the spraypaint
// crossfilter→query loop. Import from "@buhera/spraypaint/react".
//
// Requires react (>=18) and d3 (>=7) as peer dependencies. These are only pulled
// in when you import this subpath; the core "@buhera/spraypaint" stays dep-free.

export { useSpraypaintSession, type UseSpraypaintSession } from "./useSpraypaintSession.js";
export { AllocationChart, type AllocationChartProps } from "./AllocationChart.js";
export { ResultsList, type ResultsListProps } from "./ResultsList.js";
export { IdentityBadge, type IdentityBadgeProps } from "./IdentityBadge.js";
export { SpraypaintPanel, type SpraypaintPanelProps } from "./SpraypaintPanel.js";
