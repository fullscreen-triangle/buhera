// Segmentation: Otsu threshold + level-set refinement, fuzzy membership map

import { segmentImage } from '../../mic-engine';

export interface SegmentResult {
  mask: Uint8Array;              // 1 = foreground
  membership: Float32Array;      // fuzzy [0,1] per pixel
  centroid: { x: number; y: number };
  area: number;                  // pixels
  contour: Array<[number, number]>;
}

export function segment(
  image: Float32Array,
  width: number,
  height: number,
  threshold: number,             // 0.5 = Otsu, otherwise hard threshold
): SegmentResult {
  // Use Otsu if threshold == 0.5, else use provided threshold
  const result = segmentImage({ data: image, width, height });

  // Apply threshold override: rebuild mask if not 0.5
  const mask = new Uint8Array(width * height);
  if (Math.abs(threshold - 0.5) < 1e-6) {
    mask.set(result.mask);
  } else {
    for (let i = 0; i < image.length; i++) {
      mask[i] = image[i] >= threshold ? 1 : 0;
    }
  }

  // Fuzzy membership: smooth distance from threshold
  const membership = new Float32Array(width * height);
  for (let i = 0; i < image.length; i++) {
    membership[i] = Math.max(0, Math.min(1, (image[i] - (threshold - 0.2)) / 0.4));
  }

  // Centroid of mask
  let sumX = 0, sumY = 0, area = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x]) { sumX += x; sumY += y; area++; }
    }
  }
  const centroid = area > 0
    ? { x: sumX / area, y: sumY / area }
    : { x: width / 2, y: height / 2 };

  // Simple contour: pixels where mask=1 and at least one neighbour is 0
  const contour: Array<[number, number]> = [];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      if (!mask[y * width + x]) continue;
      const isEdge =
        !mask[y * width + (x-1)] || !mask[y * width + (x+1)] ||
        !mask[(y-1) * width + x] || !mask[(y+1) * width + x];
      if (isEdge) contour.push([x, y]);
    }
  }

  return { mask, membership, centroid, area, contour };
}

// Split a single-channel image containing two distinct objects into two masks.
// Finds the two largest connected components and returns their centroids.
export function findTwoNuclei(
  image: Float32Array,
  width: number,
  height: number,
  threshold: number,
): { a: SegmentResult; b: SegmentResult } {
  // Build full foreground mask
  const fg = new Uint8Array(width * height);
  for (let i = 0; i < image.length; i++) fg[i] = image[i] >= threshold ? 1 : 0;

  // Label connected components (4-connectivity BFS)
  const labels = new Int32Array(width * height).fill(-1);
  const components: Array<{ pixels: Array<[number, number]>; size: number }> = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!fg[idx] || labels[idx] >= 0) continue;
      const label = components.length;
      const pixels: Array<[number, number]> = [];
      const queue: Array<[number, number]> = [[x, y]];
      labels[idx] = label;
      while (queue.length) {
        const [cx, cy] = queue.pop()!;
        pixels.push([cx, cy]);
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          const nx = cx+dx, ny = cy+dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const ni = ny * width + nx;
          if (!fg[ni] || labels[ni] >= 0) continue;
          labels[ni] = label;
          queue.push([nx, ny]);
        }
      }
      components.push({ pixels, size: pixels.length });
    }
  }

  // Sort by size, take the two largest
  components.sort((a, b) => b.size - a.size);
  const top2 = components.slice(0, 2);
  if (top2.length < 2) {
    // Only one region — split at x midpoint
    const mid = width / 2;
    const leftPixels = top2[0]?.pixels.filter(([x]) => x < mid) ?? [];
    const rightPixels = top2[0]?.pixels.filter(([x]) => x >= mid) ?? [];
    top2[0] = { pixels: leftPixels, size: leftPixels.length };
    top2[1] = { pixels: rightPixels, size: rightPixels.length };
  }

  function toResult(comp: { pixels: Array<[number, number]> }): SegmentResult {
    const mask = new Uint8Array(width * height);
    const membership = new Float32Array(width * height);
    let sumX = 0, sumY = 0;
    for (const [px, py] of comp.pixels) {
      mask[py * width + px] = 1;
      membership[py * width + px] = image[py * width + px];
      sumX += px; sumY += py;
    }
    const area = comp.pixels.length;
    const centroid = area > 0 ? { x: sumX / area, y: sumY / area } : { x: 0, y: 0 };
    // contour
    const contour: Array<[number, number]> = [];
    for (const [px, py] of comp.pixels) {
      const isEdge =
        px === 0 || px === width-1 || py === 0 || py === height-1 ||
        !mask[py * width + (px-1)] || !mask[py * width + (px+1)] ||
        !mask[(py-1) * width + px] || !mask[(py+1) * width + px];
      if (isEdge) contour.push([px, py]);
    }
    return { mask, membership, centroid, area, contour };
  }

  return { a: toResult(top2[0] ?? { pixels: [] }), b: toResult(top2[1] ?? { pixels: [] }) };
}
