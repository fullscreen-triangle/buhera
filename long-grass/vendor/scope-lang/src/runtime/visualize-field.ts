/**
 * Visualize Coordinate Field Φ
 * Render as canvas showing partition structure and measurements
 */

import { CoordinateField } from './spectral-pipeline';

export interface VisualizationOptions {
  width?: number;
  height?: number;
  showGrid?: boolean;
  showMeasurements?: boolean;
  programName?: string;
  hasMeasure?: boolean;
  measurements?: Array<{
    label: string;
    pixel_a: { u: number; v: number };
    pixel_b: { u: number; v: number };
    distance_um: number;
  }>;
}

export function visualizeCoordinateField(
  φ: CoordinateField,
  canvas: HTMLCanvasElement,
  options: VisualizationOptions = {}
): void {
  const width = options.width || 512;
  const height = options.height || 512;
  const showGrid = options.showGrid !== false;
  const showMeasurements = options.showMeasurements !== false;
  const programName = options.programName || 'Unknown';
  const hasMeasure = options.hasMeasure || false;

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Background
  ctx.fillStyle = '#1e1e1e';
  ctx.fillRect(0, 0, width, height);

  // Render coordinate field as color gradient
  // Color based on distance from center (showing field structure)
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  const centerX = φ.field_width_um / 2;
  const centerY = φ.field_height_um / 2;
  const maxDist = Math.sqrt(centerX * centerX + centerY * centerY);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;

      // Map pixel to world-space
      const worldPos = φ.φ(x, y);

      // Compute distance from center in world-space
      const dx = worldPos.x - centerX;
      const dy = worldPos.y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(dist / maxDist, 1);

      // Color gradient varies based on whether measurements are present
      let hue: number;
      if (hasMeasure) {
        // For measurement programs: blue center → cyan → green → yellow → red edge
        hue = (1 - normalized) * 300; // 300=blue, 0=red in HSL
      } else {
        // For observation programs: blue center → purple → red edge
        hue = (1 - normalized) * 240; // 240=blue, 0=red in HSL
      }

      const rgb = hslToRgb(hue, 100, 50);

      data[idx] = rgb[0];     // R
      data[idx + 1] = rgb[1]; // G
      data[idx + 2] = rgb[2]; // B
      data[idx + 3] = 200;    // A (alpha)
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Draw grid (µm scale)
  if (showGrid) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    const gridSpacing = φ.field_width_um / 4; // Show 4x4 grid
    const pixelPerMicron = width / φ.field_width_um;

    for (let um = gridSpacing; um < φ.field_width_um; um += gridSpacing) {
      const px = um * pixelPerMicron;

      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();

      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(0, px);
      ctx.lineTo(width, px);
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    for (let um = gridSpacing; um < φ.field_width_um; um += gridSpacing) {
      const px = um * pixelPerMicron;
      ctx.fillText(`${um.toFixed(0)}µm`, px - 2, 15);
    }
  }

  // Draw measurements
  if (showMeasurements && options.measurements && options.measurements.length > 0) {
    for (const meas of options.measurements) {
      const pa = meas.pixel_a;
      const pb = meas.pixel_b;

      // Draw line between points
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pa.u, pa.v);
      ctx.lineTo(pb.u, pb.v);
      ctx.stroke();

      // Draw circles at endpoints
      ctx.fillStyle = '#00ff00';
      ctx.beginPath();
      ctx.arc(pa.u, pa.v, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pb.u, pb.v, 6, 0, Math.PI * 2);
      ctx.fill();

      // Label with distance
      const midX = (pa.u + pb.u) / 2;
      const midY = (pa.v + pb.v) / 2;
      ctx.fillStyle = '#00ff00';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${meas.distance_um.toFixed(1)}µm`, midX, midY - 15);
    }
  } else if (!showMeasurements || !options.measurements || options.measurements.length === 0) {
    // Show placeholder for observation-only programs
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('(Coordinate Field Φ)', width / 2, height / 2);
  }

  // Draw program info banner
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(0, 0, width, 30);

  ctx.fillStyle = '#00d4ff';
  ctx.font = 'bold 12px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`Program: ${programName}`, 8, 20);

  const typeLabel = hasMeasure ? '📏 Measurement' : '👁 Observation';
  ctx.fillStyle = hasMeasure ? '#00ff00' : '#ffaa00';
  ctx.textAlign = 'right';
  ctx.fillText(typeLabel, width - 8, 20);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0,
    g = 0,
    b = 0;

  if (h < 60) {
    r = c;
    g = x;
  } else if (h < 120) {
    r = x;
    g = c;
  } else if (h < 180) {
    g = c;
    b = x;
  } else if (h < 240) {
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}
