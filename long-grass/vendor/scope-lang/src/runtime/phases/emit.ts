// Phase 4 — EMIT
// Assembles the final Result, evaluates goal criteria, verifies S-entropy sum.

import type { ScopeProgram, GoalCriterion } from '../../compiler/ast';
import type { ScopeResult, GoalStatus, ChartData, VisualData, EntropyPhasePoint } from '../result-types';
import type { MeasureOutput } from './measure';
import type { ExecuteOutput } from './execute';

export function emitPhase(
  program: ScopeProgram,
  image: Float32Array,
  width: number,
  height: number,
  measure: MeasureOutput,
  execute: ExecuteOutput,
  skC: number, stC: number, seC: number,   // entropy from COMPILE phase
  snr: number,
  crlbPixels: number,
  channelCapacity: number,
  allEntropyPoints: EntropyPhasePoint[],
  rawImageUrl?: string,
): ScopeResult {
  const log: string[] = [];

  const { sk, st, se } = execute;
  const sum = sk + st + se;
  const conserved = Math.abs(sum - 1) < 1e-10;

  log.push(
    `[EMIT]     S_k=${sk.toFixed(3)}  S_t=${st.toFixed(3)}  S_e=${se.toFixed(3)}` +
    `  sum=${sum.toFixed(15)} ${conserved ? '✓' : '⚠ VIOLATION'}`
  );

  // Goal evaluation
  const goalStatus: GoalStatus[] = [];
  const { distance, uncertainty } = execute;

  for (const c of (program.goal?.criteria ?? [])) {
    const actual = evaluateMetric(c.metric, distance, uncertainty, sk, st, se, sum, snr, channelCapacity, crlbPixels);
    const passed = compare(actual, c.op, c.threshold);
    goalStatus.push({ metric: c.metric, op: c.op, threshold: c.threshold, unit: c.unit, actual, passed });
    const icon = passed ? '✓' : '✗';
    log.push(`[GOAL]     ${c.metric}=${actual.toPrecision(4)} ${c.op} ${c.threshold} ${c.unit} ${icon}`);
  }

  // Entropy trajectory (complete)
  const entropyTrajectory: EntropyPhasePoint[] = [
    { phase: 'COMPILE', sk: skC, st: stC, se: seC },
    { phase: 'MEASURE', sk: skC, st: stC, se: seC },
    ...allEntropyPoints,
    { phase: 'EMIT', sk, st, se },
  ];

  // Chart data
  const chartData: ChartData = {
    spectralPower: measure.spectralPower,
    powerLawExponent: measure.scaleField.powerLawExponent,
    alphaMean: measure.scaleField.mean,
    scaleHistogram: measure.scaleHistogram,
    entropyTrajectory,
    uncertaintyBar: {
      d: distance ?? 0,
      deltaD: uncertainty ?? 0,
      goals: goalStatus,
    },
    channelCapacity: { snr, capacity: channelCapacity },
  };

  // Visual data
  const visualData: VisualData = {
    rawImage: image,
    rawImageUrl,
    width,
    height,
    scaleField: measure.scaleField.alpha,
    segmentationMask: execute.combinedMask,
    segmentationContour: execute.combinedContour,
    distanceMap: execute.distanceMap,
    geodesicPath: execute.geodesicPath,
    pointCloud: measure.pointCloud,
    partitionStates: execute.partitionStates,
    activeVisMode: execute.activeVisMode,
  };

  // Position: centroid of first accessed target, or midpoint
  const firstTarget = Object.values(execute.accessedTargets)[0];
  const pos: [number, number, number] = firstTarget
    ? [
        firstTarget.centroid.x * (measure.scaleField.mean),
        firstTarget.centroid.y * (measure.scaleField.mean),
        0,
      ]
    : [0, 0, 0];

  return {
    structure: program.morphisms[0]?.name ?? 'observation',
    position: pos,
    distance,
    uncertainty,
    relativeUncertainty: distance && uncertainty ? uncertainty / distance : null,
    sEntropy: { sk, st, se, sum },
    goalStatus,
    chartData,
    visualData,
    log,
    snr,
    crlbPixels,
    channelCapacity,
  };
}

function evaluateMetric(
  metric: string,
  distance: number | null,
  uncertainty: number | null,
  sk: number, st: number, se: number,
  sum: number,
  snr: number,
  channelCapacity: number,
  crlbPixels: number,
): number {
  switch (metric) {
    case 'distance_uncertainty': return uncertainty ?? Infinity;
    case 'relative_uncertainty': return (distance && uncertainty) ? uncertainty / distance : Infinity;
    case 's_entropy_conservation': return Math.abs(sum - 1);
    case 'snr': return snr;
    case 'channel_capacity': return channelCapacity;
    case 'crlb_pixels': return crlbPixels;
    default: return 0;
  }
}

function compare(actual: number, op: string, threshold: number): boolean {
  switch (op) {
    case '<':  return actual < threshold;
    case '<=': return actual <= threshold;
    case '>':  return actual > threshold;
    case '>=': return actual >= threshold;
    case '==': return Math.abs(actual - threshold) < 1e-10;
    default: return false;
  }
}
