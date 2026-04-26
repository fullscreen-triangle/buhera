// kernel.js — CMM + PSS + DIC + PVE + TEM folded into one class.

import {
  SCoord,
  sDistance,
  ternaryAddress,
  backwardNavigate,
  completionMorphism,
} from "./substrate";

function normOf(s) {
  return Math.sqrt(s.k * s.k + s.t * s.t + s.e * s.e);
}

function tierOf(s) {
  const n = normOf(s);
  if (n > 0.5) return "L1";
  if (n > 1.0) return "L2";
  if (n > 1.5) return "L3";
  return "RAM";
}

export class Kernel {
  constructor(depth = 12) {
    this.depth = depth;
    this.store = new Map();
    this.events = [];
    this.pveVerified = 0;
    this.pveRejected = 0;
    this.temSamples = 0;
  }

  allocate(coord, payload = null, metadata = {}) {
    this.pveVerify("memory_create", { coord });
    const addr = ternaryAddress(coord, this.depth);
    const obj = { coord, address: addr, tier: tierOf(coord), payload, metadata };
    this.store.set(addr, obj);
    this.temSample();
    this.events.push(`CMM.allocate addr=${addr} tier=${obj.tier}`);
    return obj;
  }

  proximity(target, k = 5) {
    const scored = [];
    for (const obj of this.store.values()) {
      scored.push([obj, sDistance(target, obj.coord)]);
    }
    scored.sort((a, b) => a[1] - b[1]);
    return scored.slice(0, k);
  }

  allObjects() {
    return [...this.store.values()];
  }

  runTrajectory(targetCoord) {
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

  surgicalRetrieve(target, k = 5) {
    const cand = this.proximity(target, k * 3);
    const results = cand.slice(0, k);
    this.events.push(`DIC.retrieve ${results.length}/${cand.length}`);
    return results;
  }

  categoricalSort(items) {
    const origin = new SCoord(0, 0, 0);
    const withDist = items.map(([c, v]) => [sDistance(origin, c), c, v]);
    withDist.sort((a, b) => a[0] - b[0]);
    this.events.push(`DIC.sort ${items.length}`);
    return withDist.map(([, c, v]) => [c, v]);
  }

  pveVerify(stmt, payload) {
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

  checkStmt(stmt, payload) {
    if (stmt === "memory_create") return payload.coord instanceof SCoord;
    if (stmt === "navigate") return payload.mode === "penultimate" || payload.mode === "explicit";
    if (stmt === "complete") return payload.s_penultimate instanceof SCoord;
    if (stmt === "resolve") return !!payload.target;
    if (stmt === "spawn") return !!payload.program;
    return true;
  }

  temSample() { this.temSamples++; }

  size() { return this.store.size; }

  stats() {
    return {
      objects: this.size(),
      pveOk: this.pveVerified,
      pveRej: this.pveRejected,
      tem: this.temSamples,
    };
  }
}
