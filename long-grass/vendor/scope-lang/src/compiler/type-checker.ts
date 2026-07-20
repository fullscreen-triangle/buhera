// SCOPE Type Checker — five invariants from the spec

import type {
  ScopeProgram, MorphismDecl, CatalyzeStep, AccessStep,
  MeasureDistanceStep, CellItem, GoalCriterion,
} from './ast';

// ── Error / Warning types (from spec) ────────────────────────────────────────

export type CompileError =
  | { kind: 'ParseError'; message: string; line: number; col: number }
  | { kind: 'DepthMismatch'; morphism: string; got: number; expected: number }
  | { kind: 'CellOverlap'; cell1: string; cell2: string; overlap: [number, number] }
  | { kind: 'EntropyBudgetExceeded'; morphism: string; weightedEpsilon: number; budget: number }
  | { kind: 'UngroundedDistance'; morphism: string; target: string }
  | { kind: 'UnknownMorphismRef'; name: string }
  | { kind: 'UnknownCellRef'; name: string }
  | { kind: 'UnknownRuleRef'; name: string };

export type CompileWarning =
  | { kind: 'GoalUnreachableAtDepth'; metric: string; requiredDepth: number;
      currentDepth: number; predictedMin: number; goalThreshold: number }
  | { kind: 'VacuousConstraint'; morphism: string; constraint: string; reason: string }
  | { kind: 'ConfidenceDiscountLarge'; morphism: string; constraint: string;
      confidence: number; effectiveEpsilon: number };

export interface TypeCheckResult {
  errors: CompileError[];
  warnings: CompileWarning[];
  /** AST with epsilon values resolved from rule table */
  program: ScopeProgram;
}

// Default ε per constraint family (overridden by rule declarations)
const DEFAULT_EPSILON: Record<string, number> = {
  conservation: 0.008,
  phase_lock: 0.010,
  thermal: 0.005,
  symmetry: 0.006,
};

const S_T_INITIAL = 0.5;
const S_E_MINIMUM = 0.1;
const ENTROPY_BUDGET = 1 - S_T_INITIAL - S_E_MINIMUM;  // 0.4

export function typeCheck(program: ScopeProgram): TypeCheckResult {
  const errors: CompileError[] = [];
  const warnings: CompileWarning[] = [];

  // Build rule lookup: "name(arg)" → epsilon
  const ruleTable = new Map<string, number>();
  for (const rule of program.rules) {
    ruleTable.set(`${rule.name}(${rule.argument})`, rule.epsilon);
  }

  // Resolve epsilon for every CatalyzeStep in-place
  const resolved = resolveEpsilons(program, ruleTable, errors);

  // Determine canonical depth
  const canonicalDepth = program.coordinateSpace?.depth ?? inferDepth(resolved);

  // ── Invariant 1: Depth Compatibility ─────────────────────────────────────
  if (canonicalDepth !== undefined) {
    for (const m of resolved.morphisms) {
      const d = m.expr.observe.depth;
      if (d !== canonicalDepth) {
        errors.push({ kind: 'DepthMismatch', morphism: m.name, got: d, expected: canonicalDepth });
      }
    }
  }

  // ── Invariant 2: Cell Partition Consistency ───────────────────────────────
  const cells = (program.channels?.items ?? []).filter((i): i is CellItem => i.kind === 'CellItem');
  for (let i = 0; i < cells.length; i++) {
    for (let j = i + 1; j < cells.length; j++) {
      const a = cells[i], b = cells[j];
      const lo = Math.max(a.boundsLow, b.boundsLow);
      const hi = Math.min(a.boundsHigh, b.boundsHigh);
      if (lo < hi) {
        errors.push({ kind: 'CellOverlap', cell1: a.name, cell2: b.name, overlap: [lo, hi] });
      }
    }
  }

  // ── Invariant 3: Entropy Budget ───────────────────────────────────────────
  for (const m of resolved.morphisms) {
    let weighted = 0;
    for (const step of m.expr.steps) {
      if (step.kind === 'CatalyzeStep') {
        const w = 1 - step.confidence * 0.5;
        weighted += step.epsilon * w;
      }
    }
    weighted += m.expr.steps.filter(s => s.kind === 'AccessStep').length * 0.05;
    if (weighted > ENTROPY_BUDGET) {
      errors.push({ kind: 'EntropyBudgetExceeded', morphism: m.name, weightedEpsilon: weighted, budget: ENTROPY_BUDGET });
    }
  }

  // ── Invariant 4: Coordinate Grounding ────────────────────────────────────
  const channelNames = new Set((program.channels?.items ?? [])
    .filter(i => i.kind === 'SyncItem').map(i => (i as any).name as string));

  for (const m of resolved.morphisms) {
    const accessed = new Set<string>(channelNames);
    for (const step of m.expr.steps) {
      if (step.kind === 'AccessStep') accessed.add(step.target);
      if (step.kind === 'MeasureDistanceStep') {
        if (!accessed.has(step.target1)) {
          errors.push({ kind: 'UngroundedDistance', morphism: m.name, target: step.target1 });
        }
        if (!accessed.has(step.target2)) {
          errors.push({ kind: 'UngroundedDistance', morphism: m.name, target: step.target2 });
        }
      }
    }
  }

  // ── Invariant 5: Goal Reachability [ext] ─────────────────────────────────
  if (program.goal && canonicalDepth !== undefined && program.coordinateSpace) {
    const { fieldX, fieldY } = program.coordinateSpace;
    const fieldSize = Math.min(fieldX, fieldY);
    const resolution = fieldSize / Math.pow(2, canonicalDepth);  // µm/step
    const alphaNominal = 1.0;  // µm/px conservative estimate

    // Min total epsilon across all morphisms
    let minEpsilonSum = 0;
    for (const m of resolved.morphisms) {
      let s = 0;
      for (const step of m.expr.steps) {
        if (step.kind === 'CatalyzeStep') s += step.epsilon;
      }
      minEpsilonSum = Math.max(minEpsilonSum, s);
    }

    const predictedMin = alphaNominal * resolution * (1 + minEpsilonSum);

    for (const c of program.goal.criteria) {
      if (c.metric === 'distance_uncertainty' && (c.op === '<' || c.op === '<=')) {
        if (predictedMin > c.threshold) {
          // Compute required depth: fieldSize / 2^d * (1 + minEps) * alpha < threshold
          const requiredDepth = Math.ceil(Math.log2(alphaNominal * fieldSize * (1 + minEpsilonSum) / c.threshold));
          warnings.push({
            kind: 'GoalUnreachableAtDepth',
            metric: c.metric,
            requiredDepth,
            currentDepth: canonicalDepth,
            predictedMin,
            goalThreshold: c.threshold,
          });
        }
      }
    }
  }

  // ── Reference checks ──────────────────────────────────────────────────────
  const morphismNames = new Set(resolved.morphisms.map(m => m.name));
  const cellNames = new Set(cells.map(c => c.name));

  if (program.dispatch) {
    for (const rule of program.dispatch.rules) {
      if (!cellNames.has(rule.cell) && cellNames.size > 0) {
        errors.push({ kind: 'UnknownCellRef', name: rule.cell });
      }
      if (rule.action.kind === 'ExecuteAction') {
        if (!morphismNames.has(rule.action.morphismRef)) {
          errors.push({ kind: 'UnknownMorphismRef', name: rule.action.morphismRef });
        }
      }
    }
  }

  // Confidence large discount warning
  for (const m of resolved.morphisms) {
    for (const step of m.expr.steps) {
      if (step.kind === 'CatalyzeStep' && step.confidence < 0.5) {
        warnings.push({
          kind: 'ConfidenceDiscountLarge',
          morphism: m.name,
          constraint: `${step.constraintName}(${step.constraintArg})`,
          confidence: step.confidence,
          effectiveEpsilon: step.epsilon * (1 - step.confidence * 0.5),
        });
      }
    }
  }

  return { errors, warnings, program: resolved };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function resolveEpsilons(program: ScopeProgram, ruleTable: Map<string, number>, errors: CompileError[]): ScopeProgram {
  const morphisms: MorphismDecl[] = program.morphisms.map(m => ({
    ...m,
    expr: {
      ...m.expr,
      steps: m.expr.steps.map(step => {
        if (step.kind !== 'CatalyzeStep') return step;
        const key = `${step.constraintName}(${step.constraintArg})`;
        let epsilon = ruleTable.get(key) ?? DEFAULT_EPSILON[step.constraintName];
        if (epsilon === undefined) {
          errors.push({ kind: 'UnknownRuleRef', name: key });
          epsilon = 0.01;
        }
        return { ...step, epsilon };
      }),
    },
  }));
  return { ...program, morphisms };
}

function inferDepth(program: ScopeProgram): number | undefined {
  for (const m of program.morphisms) {
    return m.expr.observe.depth;
  }
  return undefined;
}

// ── format helpers for console output ─────────────────────────────────────────

export function formatErrors(errors: CompileError[]): string[] {
  return errors.map(e => {
    switch (e.kind) {
      case 'ParseError':
        return `[PARSE ERROR] line ${e.line} col ${e.col}\n  ${e.message}`;
      case 'DepthMismatch':
        return `[TYPE ERROR] DepthMismatch in morphism "${e.morphism}"\n  observe n=${e.got} but coordinate_space.depth=${e.expected}`;
      case 'CellOverlap':
        return `[TYPE ERROR] CellOverlap: "${e.cell1}" and "${e.cell2}"\n  Overlap region: [${e.overlap[0]}, ${e.overlap[1]}]`;
      case 'EntropyBudgetExceeded':
        return `[TYPE ERROR] EntropyBudgetExceeded in "${e.morphism}"\n  weighted_epsilon=${e.weightedEpsilon.toFixed(4)} > budget=${e.budget.toFixed(4)}`;
      case 'UngroundedDistance':
        return `[TYPE ERROR] UngroundedDistance in "${e.morphism}": "${e.target}" not accessed before measure_distance`;
      case 'UnknownMorphismRef':
        return `[TYPE ERROR] Unknown morphism reference: "${e.name}"`;
      case 'UnknownCellRef':
        return `[TYPE ERROR] Unknown cell reference: "${e.name}"`;
      case 'UnknownRuleRef':
        return `[TYPE ERROR] Unknown rule/constraint: "${e.name}"`;
    }
  });
}

export function formatWarnings(warnings: CompileWarning[]): string[] {
  return warnings.map(w => {
    switch (w.kind) {
      case 'GoalUnreachableAtDepth':
        return `[TYPE WARNING] GoalUnreachableAtDepth\n  metric=${w.metric} threshold=${w.goalThreshold}\n  depth=${w.currentDepth} predicted_δd_min=${w.predictedMin.toFixed(4)} µm\n  Suggestion: increase depth to ${w.requiredDepth}`;
      case 'ConfidenceDiscountLarge':
        return `[TYPE WARNING] ConfidenceDiscountLarge in "${w.morphism}"\n  ${w.constraint} confidence=${w.confidence} ε_eff=${w.effectiveEpsilon.toFixed(4)}`;
      case 'VacuousConstraint':
        return `[TYPE WARNING] VacuousConstraint in "${w.morphism}": ${w.constraint} — ${w.reason}`;
    }
  });
}
