// SCOPE Runtime — orchestrates the four phases for one program + per-morphism images

import type { ScopeProgram } from '../compiler/ast';
import { compilePhase } from './phases/compile';
import { measurePhase } from './phases/measure';
import { executePhase, ExecuteOutput } from './phases/execute';
import { emitPhase } from './phases/emit';
import type { ScopeResult, EntropyPhasePoint } from './result-types';

export type { ScopeResult } from './result-types';

export interface ImagePayload {
  data: Float32Array;
  width: number;
  height: number;
}

export async function runScope(
  program: ScopeProgram,
  imagePayload: ImagePayload,
  // Optional per-morphism images. When provided, each morphism uses its own real image.
  morphismImages?: Record<string, ImagePayload>,
): Promise<ScopeResult> {
  const { data: image, width, height } = imagePayload;

  const depth = program.coordinateSpace?.depth
    ?? program.morphisms[0]?.expr.observe.depth
    ?? 10;
  const fieldSizeUm = program.coordinateSpace?.fieldX ?? 100;

  // ── Phase 1: COMPILE ───────────────────────────────────────────────────
  const compileOut = compilePhase(image, program);
  let { sk, st, se } = compileOut;

  // ── Phase 2: MEASURE (on primary image) ───────────────────────────────
  const measureOut = measurePhase(image, width, height, sk, st, se);
  sk = measureOut.entropyPoint.sk;
  st = measureOut.entropyPoint.st;
  se = measureOut.entropyPoint.se;

  // ── Phase 3: EXECUTE ───────────────────────────────────────────────────
  const morphismResults: Record<string, ExecuteOutput | null> = {};
  const allEntropyPoints: EntropyPhasePoint[] = [];
  let lastExecute: ExecuteOutput | null = null;
  let targetImage = imagePayload; // tracks the target morphism's image for EMIT

  // Determine which morphism to run
  let targetMorphismName: string | null = null;
  if (program.dispatch && compileOut.cellName) {
    const rule = program.dispatch.rules.find(r => r.cell === compileOut.cellName);
    if (rule?.action.kind === 'ExecuteAction') {
      targetMorphismName = rule.action.morphismRef;
    }
  }
  if (!targetMorphismName) {
    targetMorphismName = program.morphisms[program.morphisms.length - 1]?.name ?? null;
  }

  for (const morphism of program.morphisms) {
    const isFused = program.morphisms.some(m =>
      m.expr.steps.some(s => s.kind === 'FuseStep' && (s as any).morphismRef === morphism.name)
    );
    const isTarget = morphism.name === targetMorphismName;
    if (!isTarget && !isFused) continue;

    // Use per-morphism image if available
    const mPay  = morphismImages?.[morphism.name] ?? imagePayload;
    const mData = mPay.data;
    const mW    = mPay.width;
    const mH    = mPay.height;

    // Compute scale field for this morphism's image
    const mMeasure = (mPay !== imagePayload)
      ? measurePhase(mData, mW, mH, sk, st, se)
      : measureOut;

    const execOut = executePhase(
      morphism,
      mData, mW, mH,
      mMeasure.scaleField.alpha,
      mMeasure.scaleField.mean,
      depth, fieldSizeUm,
      sk, st, se,
      morphismResults,
    );
    morphismResults[morphism.name] = execOut;
    allEntropyPoints.push(...execOut.entropyPoints);

    if (isTarget) {
      lastExecute = execOut;
      sk = execOut.sk; st = execOut.st; se = execOut.se;
      targetImage = mPay;
    }
  }

  if (!lastExecute) {
    throw new Error('No morphism was executed');
  }

  // ── Phase 4: EMIT (using the target morphism's image) ─────────────────
  const emitMeasure = (targetImage !== imagePayload)
    ? measurePhase(targetImage.data, targetImage.width, targetImage.height,
        compileOut.sk, compileOut.st, compileOut.se)
    : measureOut;

  const result = emitPhase(
    program, targetImage.data, targetImage.width, targetImage.height,
    emitMeasure, lastExecute,
    compileOut.sk, compileOut.st, compileOut.se,
    emitMeasure.snr, emitMeasure.crlbPixels, emitMeasure.channelCapacity,
    allEntropyPoints,
    (targetImage as any).url,
  );

  result.log = [
    ...compileOut.log,
    ...measureOut.log,
    ...lastExecute.log,
    ...result.log,
  ];

  return result;
}
