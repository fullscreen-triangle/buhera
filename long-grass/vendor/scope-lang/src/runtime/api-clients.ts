// SCOPE Runtime API Clients — fetch data from external sources

export interface ImageData {
  url: string;
  data: string; // base64
  shape: [number, number];
  dtype: string;
  source: string;
}

export interface HuggingFaceModel {
  id: string;
  name: string;
  description: string;
  task: string;
  url: string;
}

export interface ReactomePathway {
  id: string;
  name: string;
  description: string;
  species: string;
  entities: number;
}

// Microscopy Database API Client
export class MicroscopyDatabaseClient {
  private static readonly BBBC_BASE = 'https://data.broadinstitute.org/bbbc';
  private static readonly CACHE_NAME = 'scope-microscopy-cache';

  static async listDatasets(): Promise<
    Array<{
      id: string;
      name: string;
      description: string;
      resolution: number;
    }>
  > {
    return [
      {
        id: 'BBBC039',
        name: 'HeLa Cells (Fluorescence)',
        description: 'Human cervical cancer cells with DAPI and Actin staining',
        resolution: 0.1,
      },
      {
        id: 'BBBC006',
        name: 'CHO Cells (Tubulin)',
        description: 'Chinese hamster ovary cells with tubulin staining',
        resolution: 0.1,
      },
      {
        id: 'BBBC008',
        name: 'Drosophila Cells',
        description: 'Fruit fly embryonic cells with multiple stains',
        resolution: 0.1,
      },
    ];
  }

  static async listImages(datasetId: string): Promise<string[]> {
    // Simulated list - in production would fetch from BBBC API
    const lists: Record<string, string[]> = {
      BBBC039: [
        'SiR_Actin_001.tif',
        'SiR_Actin_002.tif',
        'SiR_Actin_003.tif',
        'DAPI_001.tif',
        'DAPI_002.tif',
      ],
      BBBC006: ['Tubulin_001.tif', 'Tubulin_002.tif'],
      BBBC008: ['Drosophila_001.tif', 'Drosophila_002.tif'],
    };
    return lists[datasetId] || [];
  }

  static async fetchImage(datasetId: string, imageId: string): Promise<ImageData> {
    try {
      // Try to fetch from local cache first (public/datasets folder)
      const cachePath = `/datasets/${datasetId}/${imageId}`;
      const response = await fetch(cachePath, { signal: AbortSignal.timeout(5000) });

      if (response.ok) {
        const data = await response.text();
        return {
          url: cachePath,
          data,
          shape: [1024, 1024],
          dtype: 'float32',
          source: `BBBC/${datasetId} (local)`,
        };
      }

      // Try BBBC mirror if local fails
      const url = `${this.BBBC_BASE}/image_sets/${datasetId}/${imageId}`;
      const bbbc = await fetch(url, { signal: AbortSignal.timeout(5000) });

      if (bbbc.ok) {
        const data = await bbbc.text();
        return {
          url,
          data,
          shape: [1024, 1024],
          dtype: 'float32',
          source: `BBBC/${datasetId}`,
        };
      }

      throw new Error(`HTTP ${bbbc.status} from BBBC`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`Failed to fetch image: ${message}, using synthetic data`);
      const data = this.generateSyntheticImage(1024, 1024);

      return {
        url: `/datasets/${datasetId}/${imageId}`,
        data,
        shape: [1024, 1024],
        dtype: 'float32',
        source: `Synthetic (${datasetId} unavailable)`,
      };
    }
  }

  private static async getFromCache(datasetId: string, imageId: string): Promise<string | null> {
    try {
      const cache = await (self as any).caches?.open(this.CACHE_NAME);
      if (!cache) return null;

      const response = await cache.match(`${datasetId}/${imageId}`);
      if (response) {
        return await response.text();
      }
    } catch (e) {
      // Silently fail - cache not available
    }
    return null;
  }

  private static async saveToCache(datasetId: string, imageId: string, data: string): Promise<void> {
    try {
      const cache = await (self as any).caches?.open(this.CACHE_NAME);
      if (!cache) return;

      const response = new Response(data, { headers: { 'Content-Type': 'application/octet-stream' } });
      await cache.put(`${datasetId}/${imageId}`, response);
    } catch (e) {
      // Silently fail - cache not available
    }
  }

  private static generateSyntheticImage(width: number, height: number): string {
    // Generate a synthetic float32 image (base64 encoded)
    const pixels = new Float32Array(width * height);
    for (let i = 0; i < pixels.length; i++) {
      pixels[i] = Math.random() * 255;
    }

    const bytes = new Uint8Array(pixels.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}

// HuggingFace API Client
export class HuggingFaceClient {
  private static readonly API_BASE = 'https://huggingface.co/api';
  private static readonly HF_TOKEN = process.env.NEXT_PUBLIC_HF_TOKEN || '';

  static async searchModels(query: string): Promise<HuggingFaceModel[]> {
    try {
      const response = await fetch(
        `${this.API_BASE}/models?search=${encodeURIComponent(query)}&limit=10`,
        {
          headers: this.HF_TOKEN ? { Authorization: `Bearer ${this.HF_TOKEN}` } : {},
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return (Array.isArray(data) ? data : []).map((model: any) => ({
        id: model.id,
        name: model.id.split('/')[1],
        description: model.description || '',
        task: model.tags?.[0] || 'unknown',
        url: `https://huggingface.co/${model.id}`,
      }));
    } catch (error) {
      console.error('HuggingFace search failed:', error);
      return [];
    }
  }

  static async getModel(modelId: string): Promise<HuggingFaceModel | null> {
    try {
      const response = await fetch(`${this.API_BASE}/models/${modelId}`, {
        headers: this.HF_TOKEN ? { Authorization: `Bearer ${this.HF_TOKEN}` } : {},
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return {
        id: data.id,
        name: data.id.split('/')[1],
        description: data.description || '',
        task: data.tags?.[0] || 'unknown',
        url: `https://huggingface.co/${data.id}`,
      };
    } catch (error) {
      console.error('HuggingFace fetch failed:', error);
      return null;
    }
  }
}

// Reactome API Client
export class ReactomeClient {
  private static readonly API_BASE = 'https://reactome.org/ContentService/data';

  static async searchPathways(query: string): Promise<ReactomePathway[]> {
    try {
      const response = await fetch(
        `${this.API_BASE}/search?query=${encodeURIComponent(query)}&species=Homo%20sapiens&limit=10`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return (data.results || [])
        .filter((r: any) => r.type === 'Pathway')
        .map((pathway: any) => ({
          id: pathway.stId,
          name: pathway.name,
          description: pathway.description || '',
          species: 'Homo sapiens',
          entities: 0,
        }));
    } catch (error) {
      console.error('Reactome search failed:', error);
      return [];
    }
  }

  static async getPathway(pathwayId: string): Promise<ReactomePathway | null> {
    try {
      const response = await fetch(`${this.API_BASE}/pathway/${pathwayId}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return {
        id: data.stId,
        name: data.displayName,
        description: data.summation?.[0]?.text || '',
        species: data.species?.[0]?.displayName || 'Unknown',
        entities: data.hasEvent?.length || 0,
      };
    } catch (error) {
      console.error('Reactome fetch failed:', error);
      return null;
    }
  }
}
