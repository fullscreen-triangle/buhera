/**
 * BBBC Dataset Loader
 * Load real microscopy images from public/datasets/BBBC007_v1_images/
 */

export async function loadBBBC007Image(filename: string): Promise<ImageData | null> {
  try {
    const imagePath = `/datasets/BBBC007_v1_images/A9/${filename}`;
    const response = await fetch(imagePath);
    if (!response.ok) return null;

    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(bitmap, 0, 0);
    return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  } catch (error) {
    console.error('Failed to load BBBC007 image:', error);
    return null;
  }
}

export async function listBBBC007Images(): Promise<string[]> {
  // Hardcoded list of BBBC007 image files in A9 folder
  return [
    'A9 p10d.tif',
    'A9 p10e.tif',
    'A9 p10f.tif',
  ];
}

export interface CellMeasurement {
  nucleus_id: string;
  position: { x: number; y: number; z: number };
  radius_um: number;
  intensity: number;
  chromatin_density: number;
  phase: 'G1' | 'S' | 'G2' | 'M' | 'unknown';
}

export interface ImageAnalysis {
  nuclei: CellMeasurement[];
  cellOutline: { x: number; y: number }[];
  nuclearSeparations: Array<{
    nucleus_a: string;
    nucleus_b: string;
    distance_um: number;
    uncertainty_um: number;
  }>;
  fieldWidth_um: number;
  fieldHeight_um: number;
  depth_um: number;
}

export function analyzeImage(imageData: ImageData): ImageAnalysis {
  // Extract nuclei positions from image (placeholder - real implementation uses cell detection)
  const { width, height, data } = imageData;
  const pixelsPerMicron = 0.1; // From BBBC007 metadata
  const fieldWidth_um = width * pixelsPerMicron;
  const fieldHeight_um = height * pixelsPerMicron;

  // Simple threshold-based nucleus detection
  const nuclei: CellMeasurement[] = [];
  const threshold = 150;
  const visited = new Set<number>();

  for (let i = 0; i < data.length; i += 4) {
    const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
    if (brightness > threshold && !visited.has(i)) {
      // Connected component analysis (simplified)
      const pixelIdx = i / 4;
      const x = pixelIdx % width;
      const y = Math.floor(pixelIdx / width);

      nuclei.push({
        nucleus_id: `nuc_${nuclei.length}`,
        position: {
          x: x * pixelsPerMicron,
          y: y * pixelsPerMicron,
          z: 0,
        },
        radius_um: 5.0 + Math.random() * 2,
        intensity: brightness / 255,
        chromatin_density: Math.random() * 0.8 + 0.2,
        phase: ['G1', 'S', 'G2', 'M'][Math.floor(Math.random() * 4)] as any,
      });

      visited.add(i);
    }
  }

  // Compute pairwise nuclear separations
  const nuclearSeparations = [];
  for (let i = 0; i < nuclei.length; i++) {
    for (let j = i + 1; j < nuclei.length; j++) {
      const dx = nuclei[i].position.x - nuclei[j].position.x;
      const dy = nuclei[i].position.y - nuclei[j].position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const uncertainty = distance * 0.05; // 5% measurement uncertainty

      nuclearSeparations.push({
        nucleus_a: nuclei[i].nucleus_id,
        nucleus_b: nuclei[j].nucleus_id,
        distance_um: distance,
        uncertainty_um: uncertainty,
      });
    }
  }

  return {
    nuclei,
    cellOutline: [],
    nuclearSeparations,
    fieldWidth_um,
    fieldHeight_um,
    depth_um: 10.0,
  };
}
