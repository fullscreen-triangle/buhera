// kernel.ts — the five subsystems in one file for browser simplicity.
// CMM (memory), PSS (scheduler), DIC (retrieval), PVE (verification), TEM (monitoring)

import {
  SCoord,
  sDistance,
  ternaryAddress,
  backwardNavigate,
  completionMorphism,
} from "./substrate";

export interface MemoryObject {
  coord: SCoord;
  address: string;
  tier: string;
  payload: unknown;
  metadata: Record<string, unknown>;
}

function normOf(s: SCoord): number {
  return Math.sqrt(s.k * s.k + s.t * s.t + s.e * s.e);
}

function tierOf(s: SCoord): string {
  const n = normOf(s);
  if (n > 0.5) return "L1";
  if (n > 1.0) return "L2";
  if (n > 1.5) return "L3";
  return "RAM";
}

// ── Kernel ────────────────────────────────────────────────

export class Kernel {
  depth: number;
  private store = new Map<string, MemoryObject>();
  private events: string[] = [];
  pveVerified = 0;
  pveRejected = 0;
  temSamples = 0;

  constructor(depth = 12) { this.depth = depth; }

  // ── CMM ──────────────
  allocate(coord: SCoord, payload: unknown = null, metadata: Record<string, unknown> = {}): MemoryObject {
    this.pveVerify("memory_create", { coord });
    const addr = ternaryAddress(coord, this.depth);
    const obj: MemoryObject = { coord, address: addr, tier: tierOf(coord), payload, metadata };
    this.store.set(addr, obj);
    this.temSample();
    this.events.push(`CMM.allocate addr=${addr} tier=${obj.tier}`);
    return obj;
  }

  proximity(target: SCoord, k = 5): Array<[MemoryObject, number]> {
    const scored: Array<[MemoryObject, number]> = [];
    for (const obj of this.store.values()) {
      scored.push([obj, sDistance(target, obj.coord)]);
    }
    scored.sort((a, b) => a[1] - b[1]);
    return scored.slice(0, k);
  }

  allObjects(): MemoryObject[] {
    return [...this.store.values()];
  }

  // ── PSS + substrate wrapper ─────
  runTrajectory(targetCoord: SCoord): { penultimate: SCoord; final: SCoord; steps: number } {
    this.pveVerify("navigate", { mode: "penultimate" });
    const traj = backwardNavigate(targetCoord, this.depth);
    const pen = traj.path.length >= 2 ? traj.path[traj.path.length - 2] : traj.path[0];
    this.temSample();
    this.pveVerify("complete", { s_penultimate: pen });
    const final = completionMorphism(pen, targetCoord);
    this.temSample();
    this.events.push(`PSS.trajectory steps=${traj.steps}`);
    return { penultimate: pen, final, steps: traj.steps };
  }

  // ── DIC ──────────────
  surgicalRetrieve(target: SCoord, k = 5): Array<[MemoryObject, number]> {
    const candidates = this.proximity(target, k * 3);
    const results = candidates.slice(0, k);
    this.events.push(`DIC.retrieve ${results.length}/${candidates.length}`);
    return results;
  }

  categoricalSort(items: Array<[SCoord, unknown]>): Array<[SCoord, unknown]> {
    const origin = new SCoord(0, 0, 0);
    const withDist = items.map(([c, v]) => [sDistance(origin, c), c, v] as const);
    withDist.sort((a, b) => a[0] - b[0]);
    this.events.push(`DIC.sort ${items.length}`);
    return withDist.map(([, c, v]) => [c, v]);
  }

  // ── PVE ──────────────
  private pveVerify(stmt: string, payload: Record<string, unknown>) {
    const ok = this.checkStmt(stmt, payload);
    if (ok) {
      this.pveVerified++;
      this.events.push(`PVE.${stmt} OK`);
    } else {
      this.pveRejected++;
      this.events.push(`PVE.${stmt} REJECTED`);
      throw new Error(`PVE rejected: ${stmt}`);
    }
  }

  private checkStmt(stmt: string, payload: Record<string, unknown>): boolean {
    if (stmt === "memory_create") return payload.coord instanceof SCoord;
    if (stmt === "navigate") return payload.mode === "penultimate" || payload.mode === "explicit";
    if (stmt === "complete") return payload.s_penultimate instanceof SCoord;
    if (stmt === "resolve") return !!payload.target;
    if (stmt === "spawn") return !!payload.program;
    return true;
  }

  // ── TEM ──────────────
  private temSample() { this.temSamples++; }

  // ── diagnostics ──────
  size(): number { return this.store.size; }
  activity(): string[] { return [...this.events]; }
  stats() { return { objects: this.size(), pveOk: this.pveVerified, pveRej: this.pveRejected, tem: this.temSamples }; }
}
