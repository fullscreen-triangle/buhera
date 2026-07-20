/**
 * SCOPE Phase 3: MEASURE
 * Spectral decomposition → Coordinate Field Φ
 * Maps pixels (u,v) to world-space (x,y,z) in micrometers
 */

export interface CoordinateField {
  // Map pixel (u,v) to world-space position (µm)
  φ: (u: number, v: number) => { x: number; y: number; z: number };
  // Scale factor at pixel (u,v) - spectral metric
  α: (u: number, v: number) => number;
  // Partition depth n
  depth: number;
  // Field dimensions
  field_width_um: number;
  field_height_um: number;
}

/**
 * Generate coordinate field from synthetic data
 * Used for Examples 1-2 (basic functionality)
 */
export function generateSyntheticCoordinateField(
  field_width_um: number,
  field_height_um: number,
  depth: number,
  image_width?: number,
  image_height?: number
): CoordinateField {
  // Use image dimensions if provided, otherwise assume square
  const w = image_width || 512;
  const h = image_height || 512;

  // Scale factor: µm per pixel
  const scale_x = field_width_um / w;
  const scale_y = field_height_um / h;

  return {
    φ: (u, v) => ({
      x: u * scale_x,
      y: v * scale_y,
      z: 0,
    }),
    α: (u, v) => {
      // Uniform scale factor (spectral metric)
      return (scale_x + scale_y) / 2;
    },
    depth,
    field_width_um,
    field_height_um,
  };
}

/**
 * Generate coordinate field from real image
 * Placeholder for FFT + coherence enforcement
 * (Full implementation requires FFT.js and spectral analysis)
 */
export function generateCoordinateFieldFromImage(
  imageData: ImageData,
  field_width_um: number,
  field_height_um: number,
  depth: number
): CoordinateField {
  const w = imageData.width;
  const h = imageData.height;

  // TODO: Implement actual spectral decomposition
  // For now, use simple linear mapping (synthetic behavior on real data)
  const scale_x = field_width_um / w;
  const scale_y = field_height_um / h;

  // Extract mean intensity for scale estimation (placeholder)
  const pixels = imageData.data;
  let meanIntensity = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    meanIntensity += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
  }
  meanIntensity /= pixels.length / 4;

  // Scale factor influenced by image content (very simple heuristic)
  const contentFactor = 0.8 + (meanIntensity / 255) * 0.4;

  return {
    φ: (u, v) => ({
      x: u * scale_x,
      y: v * scale_y,
      z: 0,
    }),
    α: (u, v) => {
      return ((scale_x + scale_y) / 2) * contentFactor;
    },
    depth,
    field_width_um,
    field_height_um,
  };
}

/**
 * Compute world-space distance between two pixel coordinates
 * Uses coordinate field Φ to map pixels to µm
 */
export function measureDistance(
  φ: CoordinateField,
  pixel_a: { u: number; v: number },
  pixel_b: { u: number; v: number }
): { distance_um: number; uncertainty_um: number } {
  const pos_a = φ.φ(pixel_a.u, pixel_a.v);
  const pos_b = φ.φ(pixel_b.u, pixel_b.v);

  const dx = pos_b.x - pos_a.x;
  const dy = pos_b.y - pos_a.y;
  const dz = pos_b.z - pos_a.z;

  const distance_um = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Uncertainty from spectral metric resolution
  // Proportional to scale factor and depth
  const scale_a = φ.α(pixel_a.u, pixel_a.v);
  const scale_b = φ.α(pixel_b.u, pixel_b.v);
  const mean_scale = (scale_a + scale_b) / 2;

  // Uncertainty ≈ scale_factor × field_resolution / depth
  const uncertainty_um = (mean_scale * Math.sqrt(2)) / Math.sqrt(φ.depth);

  return { distance_um, uncertainty_um };
}
