// SCOPE AST node types
// Extensions beyond the published paper are marked [ext]

export interface ScopeProgram {
  kind: 'ScopeProgram';
  name: string;
  channels?: ChannelsDecl;
  coordinateSpace?: CoordinateSpaceDecl;
  goal?: GoalDecl;           // [ext]
  rules: RuleDecl[];         // [ext]
  morphisms: MorphismDecl[];
  dispatch?: DispatchDecl;
}

// ── channels ────────────────────────────────────────────────────────────────

export interface ChannelsDecl { kind: 'ChannelsDecl'; items: ChannelItem[]; }
export type ChannelItem = SyncItem | CellItem;

export interface SyncItem {
  kind: 'SyncItem'; name: string; value: number; unit: string;
}
export interface CellItem {
  kind: 'CellItem'; name: string;
  boundsLow: number; boundsHigh: number; action: string;
}

// ── coordinate_space ─────────────────────────────────────────────────────────

export interface CoordinateSpaceDecl {
  kind: 'CoordinateSpaceDecl';
  fieldX: number; fieldY: number; unit: string;
  depth: number; lambdaS: number; lambdaT: number;
}

// ── goal [ext] ────────────────────────────────────────────────────────────────

export interface GoalDecl { kind: 'GoalDecl'; criteria: GoalCriterion[]; }
export interface GoalCriterion {
  kind: 'GoalCriterion';
  metric: string;
  op: '<' | '<=' | '>' | '>=' | '==';
  threshold: number;
  unit: string;
}

// ── rule [ext] ────────────────────────────────────────────────────────────────

export interface RuleDecl {
  kind: 'RuleDecl';
  name: string;
  argument: string;
  invariant: string;
  epsilon: number;
}

// ── morphisms ─────────────────────────────────────────────────────────────────

export interface MorphismDecl {
  kind: 'MorphismDecl'; name: string; expr: MorphismExpr;
}
export interface MorphismExpr {
  kind: 'MorphismExpr'; observe: ObserveExpr; steps: MorphismStep[];
}
export interface ObserveExpr {
  kind: 'ObserveExpr'; frame: FrameRef; depth: number;
}

export type FrameRef =
  | { kind: 'ChannelRef'; name: string }
  | { kind: 'LoadRef'; db: string; dataset: string; image: string };

export type MorphismStep =
  | CatalyzeStep | FuseStep | MeasureDistanceStep | AccessStep | VisualiseStep;

export interface CatalyzeStep {
  kind: 'CatalyzeStep';
  constraintName: string;    // e.g. "conservation"
  constraintArg: string;     // e.g. "dna_mass"
  epsilon: number;           // resolved at type-check time
  confidence: number;        // [ext] default 1.0
}
export interface FuseStep {
  kind: 'FuseStep'; morphismRef: string; rho: number;
}
export interface MeasureDistanceStep {
  kind: 'MeasureDistanceStep'; target1: string; target2: string;
}
export interface AccessStep {
  kind: 'AccessStep'; target: string;
  threshold: number;         // [ext] default 0.5
}
export interface VisualiseStep {
  kind: 'VisualiseStep'; mode: VisMode; // [ext]
}

export type VisMode =
  | 'raw_image' | 'scale_field' | 'segmentation' | 'distance_map' | 'geodesic'
  | 'point_cloud' | 'entropy_sphere' | 'partition_tree' | 'distance_tube'
  | 'spectral_power' | 'entropy_trajectory' | 'uncertainty_bar'
  | 'scale_histogram';

// ── dispatch ──────────────────────────────────────────────────────────────────

export interface DispatchDecl { kind: 'DispatchDecl'; rules: WhenStmt[]; }
export interface WhenStmt { kind: 'WhenStmt'; cell: string; action: Action; }
export type Action =
  | { kind: 'ExecuteAction'; morphismRef: string }
  | { kind: 'EmitAction'; name: string }
  | { kind: 'BlockAction'; actions: Action[] };
