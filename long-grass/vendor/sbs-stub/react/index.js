/* @sachikonye/sbs/react — stub.
 *
 * MetricsDashboard renders a placeholder note explaining that the real
 * dashboard from the SBS package is not installed. The Buhera terminal
 * still compiles; the SBS module returns "runtime-missing" ActResults at
 * dispatch time.
 */

import React from "react";

export function MetricsDashboard(_props) {
  return React.createElement(
    "div",
    { className: "text-xs text-gray-500 italic" },
    "SBS MetricsDashboard placeholder — real @sachikonye/sbs package not installed."
  );
}
