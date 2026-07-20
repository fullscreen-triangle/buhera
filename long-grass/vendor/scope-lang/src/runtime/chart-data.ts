/**
 * Chart data structures for visualization
 */

export interface ChartData {
  type: 'entropy' | 'measurement' | 'uncertainty';
  title: string;
  data: any;
}

export interface EntropyChart {
  type: 'entropy';
  title: 'S-Entropy Conservation';
  phases: Array<{
    phase: string;
    S_k: number;
    S_t: number;
    S_e: number;
  }>;
}

export interface MeasurementChart {
  type: 'measurement';
  title: 'Distance Measurements';
  measurements: Array<{
    label: string;
    distance_um: number;
    uncertainty_um: number;
  }>;
}

export function generateEntropyChart(s_k: number, s_t: number, s_e: number): EntropyChart {
  return {
    type: 'entropy',
    title: 'S-Entropy Conservation',
    phases: [
      { phase: 'COMPILE', S_k: 0.25, S_t: 0.50, S_e: 0.25 },
      { phase: 'ASSIGN', S_k: 0.30, S_t: 0.45, S_e: 0.25 },
      { phase: 'MEASURE', S_k: 0.32, S_t: 0.44, S_e: 0.24 },
      { phase: 'EXECUTE', S_k: s_k, S_t: s_t, S_e: s_e },
    ],
  };
}

export function generateMeasurementChart(
  distance?: number,
  uncertainty?: number
): MeasurementChart {
  return {
    type: 'measurement',
    title: 'Distance Measurements',
    measurements: distance
      ? [
          {
            label: 'nucleus_a → nucleus_b',
            distance_um: distance,
            uncertainty_um: uncertainty || 0,
          },
        ]
      : [],
  };
}
