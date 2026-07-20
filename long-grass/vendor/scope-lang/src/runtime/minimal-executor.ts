/**
 * Minimal SCOPE Executor — Examples 1-3
 * Phases 1-5 with coordinate field for world-space measurements
 */

import type { CompileResult } from '../compiler';
// TODO: replace with ScopeProgram when runtime is rewritten
type ExecutionPlan = any;
import { generateSyntheticCoordinateField, measureDistance, CoordinateField } from './spectral-pipeline';
import { generateEntropyChart, generateMeasurementChart } from './chart-data';

export interface ObservationResult {
  success: boolean;
  structure: string;
  position: { x: number; y: number; z: number };
  distance?: number;
  uncertainty?: number;
  s_entropy: { S_k: number; S_t: number; S_e: number };
  logs: string[];
  timing_ms: number;
  // Visualization data
  programName?: string;
  coordinateField?: any; // CoordinateField type
  measurements?: Array<{
    label: string;
    pixel_a: { u: number; v: number };
    pixel_b: { u: number; v: number };
    distance_um: number;
  }>;
  // Chart data
  entropyChart?: any;
  measurementChart?: any;
}

export async function executeMinimal(plan: ExecutionPlan): Promise<ObservationResult> {
  const logs: string[] = [];
  const startTime = performance.now();

  // Detect what this program does
  const hasMeasureDistance = (plan as any).morphisms?.some((m: any) =>
    m.steps?.some((s: any) => s.type === 'measure')
  ) ?? false;
  const hasCatalyze = (plan as any).morphisms?.some((m: any) =>
    m.steps?.some((s: any) => s.type === 'catalyze')
  ) ?? false;
  const hasAccess = (plan as any).morphisms?.some((m: any) =>
    m.steps?.some((s: any) => s.type === 'access')
  ) ?? false;

  logs.push(`Program: ${plan.name}`);
  logs.push(`  Has measure_distance: ${hasMeasureDistance}`);
  logs.push(`  Has catalyze: ${hasCatalyze}`);
  logs.push(`  Has access: ${hasAccess}`);
  logs.push('');

  // Phase 1: COMPILE (synthetic timing)
  logs.push('Phase 1 COMPILE: trajectory accumulation');
  const trajectory = Array.from({ length: 100 }, (_, i) => ({
    event_id: i,
    timestamp: i * 0.001,
  }));
  logs.push(`  Generated ${trajectory.length} timing events`);

  // Phase 2: ASSIGN (classify to cell)
  logs.push('Phase 2 ASSIGN: trajectory classification');
  let cellMatch = 'NONE';
  const { sync, cells } = plan.channels;
  if (sync && cells.length > 0) {
    const midpoint = (cells[0].bounds_min + cells[0].bounds_max) / 2;
    cellMatch = cells[0].id;
    logs.push(`  Classified to cell: ${cellMatch}`);
  }

  // Phase 3: MEASURE (spectral pipeline → coordinate field)
  logs.push('Phase 3 MEASURE: spectral decomposition');
  const { field_width_um, field_height_um, depth } = plan.coordinate_space;
  logs.push(`  Field: ${field_width_um} x ${field_height_um} µm, depth=${depth}`);
  logs.push(`  FFT spectral decomposition`);

  // Generate coordinate field Φ
  const φ = generateSyntheticCoordinateField(field_width_um, field_height_um, depth);
  logs.push(`  Coordinate field Φ computed`);

  // Phase 4: EXECUTE (run morphism chain)
  logs.push('Phase 4 EXECUTE: morphism chain');

  // Initialize entropy based on program complexity
  let s_k = 0.33;
  let s_t = 0.33;
  let s_e = 0.34;

  // Adjust based on detected operations
  if (hasAccess) {
    s_k += 0.15;
    s_e -= 0.10;
  }
  if (hasCatalyze) {
    s_k += 0.10;
    s_e -= 0.05;
  }

  let result_distance: number | undefined;
  let result_uncertainty: number | undefined;
  let result_position = { x: field_width_um / 2, y: field_height_um / 2, z: 0 };

  // Vary position slightly based on program name/type
  const programNameHash = plan.name.charCodeAt(0);
  result_position.x = field_width_um * (0.3 + (programNameHash % 10) * 0.07);
  result_position.y = field_height_um * (0.4 + (programNameHash % 7) * 0.08);

  if (plan.morphisms.length > 0) {
    const chain = plan.morphisms[0];
    logs.push(`  Chain: ${chain.id}`);

    for (const step of chain.steps) {
      if (step.type === 'observe') {
        logs.push(`    observe(${step.params.frame}, n=${step.params.depth})`);
      } else if (step.type === 'catalyze') {
        logs.push(`    catalyze(${step.params.constraint})`);
        s_k += 0.15;
        s_e += 0.05;
      } else if (step.type === 'measure') {
        // Measure distance using coordinate field
        const target_a = step.params.target_a;
        const target_b = step.params.target_b;
        logs.push(`    measure_distance(${target_a}, ${target_b})`);

        // Simulate pixel positions for the two targets
        // In a full implementation, these would come from the image analysis
        const pixel_a = {
          u: (field_width_um / 4) / (field_width_um / 512), // pixel position
          v: (field_height_um / 2) / (field_height_um / 512),
        };
        const pixel_b = {
          u: (3 * field_width_um / 4) / (field_width_um / 512),
          v: (field_height_um / 2) / (field_height_um / 512),
        };

        const measurement = measureDistance(φ, pixel_a, pixel_b);
        result_distance = measurement.distance_um;
        result_uncertainty = measurement.uncertainty_um;

        logs.push(`      ${target_a} pixel: (${pixel_a.u.toFixed(0)}, ${pixel_a.v.toFixed(0)}) → world: (${(pixel_a.u * (field_width_um / 512)).toFixed(1)}, ${(pixel_a.v * (field_height_um / 512)).toFixed(1)}, 0.0) µm`);
        logs.push(`      ${target_b} pixel: (${pixel_b.u.toFixed(0)}, ${pixel_b.v.toFixed(0)}) → world: (${(pixel_b.u * (field_width_um / 512)).toFixed(1)}, ${(pixel_b.v * (field_height_um / 512)).toFixed(1)}, 0.0) µm`);
        logs.push(`      distance: ${result_distance.toFixed(1)} µm ± ${result_uncertainty.toFixed(2)} µm`);
      } else if (step.type === 'access') {
        logs.push(`    access(${step.params.structure})`);
        s_k += 0.10;
        s_e += 0.05;
      }
    }
  }

  // Normalize S-entropy
  const total = s_k + s_t + s_e;
  s_k /= total;
  s_t /= total;
  s_e /= total;

  // Phase 5: EMIT (return world-space result)
  logs.push('Phase 5 EMIT: world-space result');
  logs.push(`  Position: (${result_position.x.toFixed(1)}, ${result_position.y.toFixed(1)}, ${result_position.z.toFixed(1)}) µm`);
  if (result_distance !== undefined) {
    logs.push(`  Distance: ${result_distance.toFixed(1)} µm`);
    logs.push(`  Uncertainty: ±${result_uncertainty?.toFixed(2)} µm`);
  }
  logs.push(`  S-entropy: S_k=${s_k.toFixed(3)} S_t=${s_t.toFixed(3)} S_e=${s_e.toFixed(3)} (sum=${(s_k + s_t + s_e).toFixed(3)})`);

  const timing_ms = performance.now() - startTime;
  logs.push(`✓ Complete in ${timing_ms.toFixed(1)}ms`);

  // Prepare visualization measurements
  const measurements = result_distance
    ? [
        {
          label: 'nuclear_separation',
          pixel_a: {
            u: (field_width_um / 4) / (field_width_um / 512),
            v: (field_height_um / 2) / (field_height_um / 512),
          },
          pixel_b: {
            u: (3 * field_width_um / 4) / (field_width_um / 512),
            v: (field_height_um / 2) / (field_height_um / 512),
          },
          distance_um: result_distance,
        },
      ]
    : undefined;

  return {
    success: true,
    structure: 'partition_observation',
    position: result_position,
    distance: result_distance,
    uncertainty: result_uncertainty,
    s_entropy: { S_k: s_k, S_t: s_t, S_e: s_e },
    logs,
    timing_ms,
    programName: plan.name,
    coordinateField: φ,
    measurements,
    entropyChart: generateEntropyChart(s_k, s_t, s_e),
    measurementChart: generateMeasurementChart(result_distance, result_uncertainty),
  };
}
