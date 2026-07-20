// Shared result types for the SCOPE runtime

export interface GoalStatus {
  metric: string;
  op: string;
  threshold: number;
  unit: string;
  actual: number;
  passed: boolean;
}

export interface SpectralPowerPoint { freq: number; energy: number; }
export interface ScaleHistogramBin  { alpha: number; count: number; }
export interface EntropyPhasePoint  { phase: string; sk: number; st: number; se: number; }

export interface ChartData {
  spectralPower: SpectralPowerPoint[];
  powerLawExponent: number;
  alphaMean: number;            // mean of α(x,y) scale field — for ScaleHistogram mean line
  scaleHistogram: ScaleHistogramBin[];
  entropyTrajectory: EntropyPhasePoint[];
  uncertaintyBar: { d: number; deltaD: number; goals: GoalStatus[] };
  channelCapacity: { snr: number; capacity: number };
}

export interface VisualData {
  rawImage: Float32Array;
  rawImageUrl?: string;   // color JPG URL for direct browser rendering
  width: number;
  height: number;
  scaleField: Float32Array;        // α(x,y)
  segmentationMask: Uint8Array;    // combined mask of all accessed targets
  segmentationContour: Array<[number, number]>;
  distanceMap: Float32Array | null;
  geodesicPath: Array<[number, number]>;
  pointCloud: Float32Array | null; // [x,y,z, r,g,b] × N, lazy-built
  partitionStates: PartitionStateNode[];
  activeVisMode: string | null;    // last visualise() mode hint
}

export interface PartitionStateNode {
  id: string;
  label: string;
  n: number; l: number; m: number; s: number;
  sk: number; st: number; se: number;
  parentId: string | null;
}

export interface ScopeResult {
  structure: string;
  position: [number, number, number];
  distance: number | null;
  uncertainty: number | null;
  relativeUncertainty: number | null;
  sEntropy: { sk: number; st: number; se: number; sum: number };
  goalStatus: GoalStatus[];
  chartData: ChartData;
  visualData: VisualData;
  log: string[];
  snr: number;
  crlbPixels: number;
  channelCapacity: number;
}
