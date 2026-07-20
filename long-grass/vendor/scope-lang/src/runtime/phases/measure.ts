// Phase 2 — MEASURE
// Runs the three-stage MIC spectral pipeline → α(x,y), chart data, point cloud.
// Entropy is unchanged (MEASURE is a deterministic bijection).

import { computeScaleField } from '../mic/scale-field';
import { computeEntropy } from '../mic/entropy';
import type { ScaleField, SpectralPowerPoint, ScaleHistogramBin } from '../mic/scale-field';
import type { EntropyPhasePoint } from '../result-types';

export interface MeasureOutput {
  scaleField: ScaleField;
  spectralPower: SpectralPowerPoint[];
  scaleHistogram: ScaleHistogramBin[];
  channelCapacity: number;
  snr: number;
  crlbPixels: number;
  shannonH: number;
  pointCloud: Float32Array;   // [x,y,z, r,g,b] × N (subsampled to ≤50k pts)
  entropyPoint: EntropyPhasePoint;
  log: string[];
}

// viridis palette: 8 control points → interpolate
const VIRIDIS: Array<[number,number,number]> = [
  [68,1,84],[72,40,120],[62,74,137],[49,104,142],
  [38,130,142],[31,158,137],[53,183,121],[110,206,88],[180,222,44],[253,231,37],
];
function viridis(t: number): [number,number,number] {
  const scaled = t * (VIRIDIS.length - 1);
  const lo = Math.floor(scaled), hi = Math.min(VIRIDIS.length - 1, lo + 1);
  const f = scaled - lo;
  return [
    VIRIDIS[lo][0] + f * (VIRIDIS[hi][0] - VIRIDIS[lo][0]),
    VIRIDIS[lo][1] + f * (VIRIDIS[hi][1] - VIRIDIS[lo][1]),
    VIRIDIS[lo][2] + f * (VIRIDIS[hi][2] - VIRIDIS[lo][2]),
  ];
}

export function measurePhase(
  image: Float32Array,
  width: number,
  height: number,
  sk: number, st: number, se: number,
): MeasureOutput {
  const log: string[] = [];

  const sfOut = computeScaleField(image, width, height);
  const { field } = sfOut;

  log.push(
    `[MEASURE]  ᾱ=${field.mean.toFixed(3)} µm/px  σ_α=${field.stddev.toFixed(3)}` +
    `  power_law=${field.powerLawExponent.toFixed(3)}` +
    `  C=${sfOut.channelCapacity.toFixed(2)} bits/px  bilateral ✓`
  );

  const entropy = computeEntropy(image, width, height);

  // Build point cloud (subsample every N pixels to keep ≤50k)
  const total = width * height;
  const step = Math.max(1, Math.ceil(total / 50000));
  const nPts = Math.ceil(total / step);
  const pc = new Float32Array(nPts * 6);

  // normalise alpha for colour mapping
  let alphaMin = Infinity, alphaMax = -Infinity;
  for (let i = 0; i < field.alpha.length; i++) {
    if (field.alpha[i] < alphaMin) alphaMin = field.alpha[i];
    if (field.alpha[i] > alphaMax) alphaMax = field.alpha[i];
  }
  const alphaRange = Math.max(0.001, alphaMax - alphaMin);

  let ptIdx = 0;
  for (let i = 0; i < total; i += step) {
    const x = i % width;
    const y = Math.floor(i / width);
    const z = image[i] * 10;  // intensity → z height (µm scale)
    const t = (field.alpha[i] - alphaMin) / alphaRange;
    const [r, g, b] = viridis(t);
    pc[ptIdx * 6 + 0] = x;
    pc[ptIdx * 6 + 1] = y;
    pc[ptIdx * 6 + 2] = z;
    pc[ptIdx * 6 + 3] = r / 255;
    pc[ptIdx * 6 + 4] = g / 255;
    pc[ptIdx * 6 + 5] = b / 255;
    ptIdx++;
  }

  // Entropy unchanged at MEASURE (deterministic)
  const entropyPoint: EntropyPhasePoint = { phase: 'MEASURE', sk, st, se };

  return {
    scaleField: field,
    spectralPower: sfOut.spectralPower,
    scaleHistogram: sfOut.scaleHistogram,
    channelCapacity: sfOut.channelCapacity,
    snr: sfOut.snr,
    crlbPixels: entropy.crlbPixels,
    shannonH: entropy.shannonH,
    pointCloud: pc,
    entropyPoint,
    log,
  };
}
