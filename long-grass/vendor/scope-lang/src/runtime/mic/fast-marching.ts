// Fast Marching geodesic distance + path backtrack
// Uses the existing mic-engine implementation

import { fastMarchingDistance } from '../../mic-engine';

export interface GeodesicResult {
  distanceMap: Float32Array;   // T(x,y) from source
  distance: number;            // T at target pixel
  path: Array<[number, number]>; // pixel coords from source to target
  pathLengthPx: number;
}

export function geodesicDistance(
  alpha: Float32Array,
  width: number,
  height: number,
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
): GeodesicResult {
  const distanceMap = fastMarchingDistance(
    alpha, width, height,
    Math.round(sourceX), Math.round(sourceY),
  );
  const distance = distanceMap[Math.round(targetY) * width + Math.round(targetX)];

  // Backtrack gradient descent on T(x,y) from target to source
  const path: Array<[number, number]> = [];
  let cx = Math.round(targetX);
  let cy = Math.round(targetY);
  const maxSteps = width * height;

  for (let step = 0; step < maxSteps; step++) {
    path.push([cx, cy]);
    if (Math.abs(cx - Math.round(sourceX)) < 2 && Math.abs(cy - Math.round(sourceY)) < 2) break;

    // Find steepest descent neighbour
    let bestVal = distanceMap[cy * width + cx];
    let bestX = cx, bestY = cy;
    for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const v = distanceMap[ny * width + nx];
      if (v < bestVal) { bestVal = v; bestX = nx; bestY = ny; }
    }
    if (bestX === cx && bestY === cy) break; // local minimum — done
    cx = bestX; cy = bestY;
  }

  return { distanceMap, distance, path, pathLengthPx: path.length };
}
