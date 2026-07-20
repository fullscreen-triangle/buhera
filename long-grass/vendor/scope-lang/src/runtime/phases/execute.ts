// Phase 3 — EXECUTE
// Walks the morphism chain step by step, updating entropy and collecting results.

import type { MorphismDecl, MorphismStep, CatalyzeStep, AccessStep,
  MeasureDistanceStep, FuseStep, VisualiseStep } from '../../compiler/ast';
import { segment, findTwoNuclei } from '../mic/segmentation';
import { geodesicDistance } from '../mic/fast-marching';
import type { SegmentResult } from '../mic/segmentation';
import type { EntropyPhasePoint, PartitionStateNode } from '../result-types';

// Standard structure → which of the two nuclei (or other region) to use
const NUCLEUS_A_NAMES = new Set(['nucleus_a', 'nucleus_centroid', 'spindle_midpoint']);
const NUCLEUS_B_NAMES = new Set(['nucleus_b', 'separation_vector', 'spindle_midpoint']);
const BOUNDARY_NAMES  = new Set(['cell_boundary', 'partition_boundary']);

export interface AccessedTarget {
  name: string;
  centroid: { x: number; y: number };
  mask: Uint8Array;
  membership: Float32Array;
  area: number;
  contour: Array<[number, number]>;
}

export interface ExecuteOutput {
  distance: number | null;
  uncertainty: number | null;
  distanceMap: Float32Array | null;
  geodesicPath: Array<[number, number]>;
  combinedMask: Uint8Array;
  combinedContour: Array<[number, number]>;
  accessedTargets: Record<string, AccessedTarget>;
  sk: number; st: number; se: number;
  activeVisMode: string | null;
  partitionStates: PartitionStateNode[];
  log: string[];
  entropyPoints: EntropyPhasePoint[];
}

export function executePhase(
  morphism: MorphismDecl,
  image: Float32Array,
  width: number,
  height: number,
  scaleField: Float32Array,
  scaleFieldMean: number,
  depth: number,
  fieldSizeUm: number,
  sk: number, st: number, se: number,
  allMorphisms: Record<string, ExecuteOutput | null>,
): ExecuteOutput {
  const log: string[] = [];
  const accessed: Record<string, AccessedTarget> = {};
  let distance: number | null = null;
  let uncertainty: number | null = null;
  let distanceMap: Float32Array | null = null;
  let geodesicPath: Array<[number, number]> = [];
  let combinedMask = new Uint8Array(width * height);
  let combinedContour: Array<[number, number]> = [];
  let activeVisMode: string | null = null;
  const partitionStates: PartitionStateNode[] = [];
  const entropyPoints: EntropyPhasePoint[] = [];

  // Running epsilon sum for uncertainty formula
  let epsilonSum = 0;

  log.push(`[ASSIGN]   morphism=${morphism.name}`);

  const n = morphism.expr.observe.depth;
  log.push(`[EXECUTE]  observe n=${n}  Σ=(${n},0,0,+½)`);

  // Root partition state
  partitionStates.push({ id: '0', label: `observe`, n, l: 0, m: 0, s: 1,
    sk, st, se, parentId: null });
  let psId = 0;

  for (const step of morphism.expr.steps) {
    switch (step.kind) {

      case 'CatalyzeStep': {
        const cs = step as CatalyzeStep;
        const w = 1 - cs.confidence * 0.5;
        const epsEff = cs.epsilon * w;
        epsilonSum += epsEff;
        sk = Math.min(0.95, sk + epsEff);
        // renormalise
        const tot = sk + st + se; sk /= tot; st /= tot; se /= tot;
        const conf = cs.confidence < 1 ? `  conf=${cs.confidence}  ε_eff=${epsEff.toFixed(4)}` : `  ε_eff=${epsEff.toFixed(4)}`;
        log.push(
          `[EXECUTE]  catalyze(${cs.constraintName}(${cs.constraintArg}))  ε=${cs.epsilon}${conf}  S_k→${sk.toFixed(3)}`
        );
        psId++;
        partitionStates.push({ id: String(psId), label: `catalyze(${cs.constraintName})`,
          n, l: 0, m: psId, s: 1, sk, st, se, parentId: String(psId - 1) });
        entropyPoints.push({ phase: `catalyze`, sk, st, se });
        break;
      }

      case 'AccessStep': {
        const as = step as AccessStep;
        const thr = as.threshold;
        sk = Math.min(0.95, sk + 0.05);
        const tot = sk + st + se; sk /= tot; st /= tot; se /= tot;

        // Decide which region to segment
        let seg: SegmentResult;
        if (NUCLEUS_A_NAMES.has(as.target) && !accessed['nucleus_a'] && !accessed[as.target]) {
          // First nucleus access — find both, assign a
          const { a, b } = findTwoNuclei(image, width, height, thr);
          // Store both even though only a is currently requested
          seg = a;
          if (!accessed['nucleus_b']) {
            accessed['nucleus_b'] = { name: 'nucleus_b', centroid: b.centroid,
              mask: b.mask, membership: b.membership, area: b.area, contour: b.contour };
          }
        } else if (NUCLEUS_B_NAMES.has(as.target) && accessed['nucleus_a'] && !accessed['nucleus_b']) {
          // Second nucleus — already pre-computed above, use it
          if (accessed['nucleus_b']) {
            seg = accessed['nucleus_b'] as unknown as SegmentResult;
          } else {
            const { b } = findTwoNuclei(image, width, height, thr);
            seg = b;
          }
        } else if (BOUNDARY_NAMES.has(as.target)) {
          // Loose boundary at lower threshold
          seg = segment(image, width, height, Math.min(thr, 0.3));
        } else {
          seg = segment(image, width, height, thr);
        }

        accessed[as.target] = {
          name: as.target,
          centroid: seg.centroid,
          mask: seg.mask,
          membership: seg.membership,
          area: seg.area,
          contour: seg.contour,
        };

        // Label each nucleus distinctly: nucleus_a / first-accessed → 1, second → 2
        const accessedNames = Object.keys(accessed);
        const labelIdx = accessedNames.indexOf(as.target);
        const maskLabel = (labelIdx <= 0) ? 1 : 2;
        for (let i = 0; i < combinedMask.length; i++) {
          if (seg.mask[i] && combinedMask[i] === 0) combinedMask[i] = maskLabel;
        }
        combinedContour.push(...seg.contour);

        log.push(
          `[EXECUTE]  access(${as.target}, threshold=${thr.toFixed(2)})` +
          `  centroid=(${Math.round(seg.centroid.x)},${Math.round(seg.centroid.y)})` +
          `  mask_area=${seg.area}px  S_k→${sk.toFixed(3)}`
        );
        psId++;
        partitionStates.push({ id: String(psId), label: `access(${as.target})`,
          n, l: 0, m: psId, s: 1, sk, st, se, parentId: String(psId - 1) });
        entropyPoints.push({ phase: `access(${as.target})`, sk, st, se });
        break;
      }

      case 'MeasureDistanceStep': {
        const ms = step as MeasureDistanceStep;
        const t1 = accessed[ms.target1];
        const t2 = accessed[ms.target2];
        if (!t1 || !t2) {
          log.push(`[EXECUTE]  measure_distance: target not found`);
          break;
        }

        log.push(`[EXECUTE]  measure_distance(${ms.target1}, ${ms.target2})  fast-marching ${width}×${height}`);

        const geo = geodesicDistance(
          scaleField, width, height,
          t1.centroid.x, t1.centroid.y,
          t2.centroid.x, t2.centroid.y,
        );

        distanceMap = geo.distanceMap;
        geodesicPath = geo.path;

        // World-space distance: use α-weighted fast marching result (already in pixel units × alpha)
        // δd = ᾱ × (fieldSize / 2^n) × (1 + Σ ε_eff)
        const resolution = fieldSizeUm / Math.pow(2, depth);
        distance = geo.distance * scaleFieldMean;   // convert px cost → µm
        uncertainty = scaleFieldMean * resolution * (1 + epsilonSum);

        log.push(
          `[EXECUTE]  d=${distance.toFixed(3)} µm  δd=${uncertainty.toFixed(3)} µm` +
          `  (${((uncertainty / distance) * 100).toFixed(2)}%)  path_length=${geo.pathLengthPx}px`
        );
        psId++;
        partitionStates.push({ id: String(psId), label: `measure_distance`,
          n, l: 0, m: psId, s: 1, sk, st, se, parentId: String(psId - 1) });
        entropyPoints.push({ phase: `measure_distance`, sk, st, se });
        break;
      }

      case 'FuseStep': {
        const fs = step as FuseStep;
        const refResult = allMorphisms[fs.morphismRef];
        if (refResult && refResult.distance !== null && distance !== null) {
          const d1: number = distance, d2: number = refResult.distance;
          const u1 = uncertainty ?? 0, u2 = refResult.uncertainty ?? 0;
          distance = fs.rho * d1 + (1 - fs.rho) * d2;
          uncertainty = Math.sqrt((fs.rho * u1) ** 2 + ((1 - fs.rho) * u2) ** 2);
          log.push(
            `[EXECUTE]  fuse(${fs.morphismRef}, rho=${fs.rho})` +
            `  d_fused=${distance.toFixed(3)} µm  δd_fused=${uncertainty.toFixed(3)} µm`
          );
        } else {
          log.push(`[EXECUTE]  fuse(${fs.morphismRef}) — ref morphism not yet executed, skipping`);
        }
        break;
      }

      case 'VisualiseStep': {
        const vs = step as VisualiseStep;
        activeVisMode = vs.mode;
        log.push(`[VISUALISE] → ${vs.mode}`);
        break;
      }
    }
  }

  return {
    distance, uncertainty, distanceMap, geodesicPath,
    combinedMask, combinedContour,
    accessedTargets: accessed,
    sk, st, se,
    activeVisMode, partitionStates,
    log, entropyPoints,
  };
}
