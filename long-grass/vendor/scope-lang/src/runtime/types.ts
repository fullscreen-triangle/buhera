// SCOPE Runtime Types

export interface PartitionState {
  n: number; // depth (spatial: field/resolution ratio; temporal: oscillator depth)
  ℓ: number; // angular mode (spatial: frequency orientation; temporal: channel index)
  m: number; // magnetic mode (spatial: frequency phase; temporal: trajectory index)
  s: 1 | -1; // spin (spatial: handedness; temporal: early/late ΔP sign)
}

export interface SEntropy {
  S_k: number; // kinetic entropy (configuration space reduction)
  S_t: number; // temporal entropy (timing accumulation)
  S_e: number; // entropic backaction (access cost)
}

export interface TimingEvent {
  delta_p: number; // ΔP = T_ref - t_rec
  channel: number; // which channel
  timestamp: number; // when it occurred
}

export interface Trajectory {
  events: TimingEvent[];
  completed: boolean;
}

export interface CoordinateField {
  field_size: [number, number]; // µm
  depth: number;
  lambda_s: number;
  lambda_t: number;
  values: Map<string, [number, number, number]>; // position -> world coordinates
}

export interface MeasurementResult {
  structure: string;
  distance: number | null; // in meters
  uncertainty: number;
  position: { x: number; y: number; z: number };
  s_entropy: SEntropy;
}

export interface ExecutionContext {
  phase: 'COMPILE' | 'ASSIGN' | 'MEASURE' | 'EXECUTE' | 'EMIT';
  timing_events: TimingEvent[];
  trajectory: Trajectory;
  cell_id: string | null;
  coord_field: CoordinateField | null;
  result: MeasurementResult | null;
  s_entropy: SEntropy;
}

export interface SCOPEProgramConfig {
  name: string;
  channels: Array<{
    id: string;
    type: 'sync' | 'cell';
    frequency?: number;
    bounds?: [number, number];
  }>;
  coordinateSpace: {
    field: [number, number, string];
    depth: number;
    lambdaS: number;
    lambdaT: number;
  };
  morphisms: Array<{
    id: string;
    steps: Array<{
      type: string;
      params: Record<string, any>;
    }>;
  }>;
  dispatchTable: Array<{
    cellId: string;
    action: string;
    chainId?: string;
    label?: string;
  }>;
}

export interface ExecutionResult {
  success: boolean;
  output: ExecutionContext;
  logs: string[];
  error?: string;
  timing_ms: number;
}
