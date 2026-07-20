// Phase 1 — COMPILE
// Derives synthetic timing events from pixel histogram, classifies into timing cell,
// updates S_t / S_k, appends entropyTrajectory[0].

import type { ScopeProgram, CellItem } from '../../compiler/ast';
import type { EntropyPhasePoint } from '../result-types';

export interface CompileOutput {
  cellName: string | null;    // which cell matched (null = default)
  sk: number;
  st: number;
  se: number;
  log: string[];
  entropyPoint: EntropyPhasePoint;
}

export function compilePhase(
  image: Float32Array,
  program: ScopeProgram,
): CompileOutput {
  const log: string[] = [];

  // Build histogram (256 bins) from image
  const bins = new Array(256).fill(0);
  for (let i = 0; i < image.length; i++) {
    bins[Math.min(255, Math.floor(image[i] * 255))]++;
  }

  // Synthetic ΔP events: map intensity bin → timing deviation
  // ΔP = (bin/255 − 0.5) × 4e-6 s
  const events: number[] = [];
  for (let b = 0; b < 256; b++) {
    if (bins[b] > 0) {
      const dp = (b / 255 - 0.5) * 4e-6;
      for (let k = 0; k < Math.min(bins[b], 50); k++) events.push(dp);
    }
  }

  const dpMean = events.reduce((a, b) => a + b, 0) / (events.length || 1);
  const dpSigma = Math.sqrt(
    events.reduce((a, b) => a + (b - dpMean) ** 2, 0) / (events.length || 1)
  );

  log.push(`[COMPILE]  events=${events.length}  ΔP_mean=${dpMean.toExponential(2)}s  σ=${dpSigma.toExponential(2)}s`);

  // Classify into timing cell
  const cells = (program.channels?.items ?? [])
    .filter((i): i is CellItem => i.kind === 'CellItem');

  let matchedCell: CellItem | null = null;
  for (const cell of cells) {
    if (dpMean >= cell.boundsLow && dpMean < cell.boundsHigh) {
      matchedCell = cell;
      break;
    }
  }

  // S-entropy update from timing classification
  let st = 0.5;
  let sk = 0.3;
  let se = 0.2;

  if (matchedCell) {
    // How narrowly the cell was classified
    const totalSpan = cells.reduce((max, c) => Math.max(max, c.boundsHigh - c.boundsLow), 0) * cells.length;
    const cellWidth = matchedCell.boundsHigh - matchedCell.boundsLow;
    const fraction = totalSpan > 0 ? cellWidth / totalSpan : 0.3;
    const deltaS_t = st * (1 - fraction);
    st -= deltaS_t;
    sk += deltaS_t * 0.8;
    se += deltaS_t * 0.2;

    log.push(
      `[COMPILE]  cell=${matchedCell.name}  bounds=(${matchedCell.boundsLow.toExponential(1)},${matchedCell.boundsHigh.toExponential(1)})` +
      `  S_t: 0.500→${st.toFixed(3)}  S_k: 0.300→${sk.toFixed(3)}`
    );
  } else {
    log.push(`[COMPILE]  default cell_label  no timing cells declared`);
  }

  // normalise
  const total = sk + st + se;
  sk /= total; st /= total; se /= total;

  const entropyPoint: EntropyPhasePoint = { phase: 'COMPILE', sk, st, se };

  return { cellName: matchedCell?.name ?? null, sk, st, se, log, entropyPoint };
}
