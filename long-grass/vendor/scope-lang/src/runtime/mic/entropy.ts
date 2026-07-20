// Entropy metrics — Shannon H, Fisher F, CRLB, SNR

import { computeEntropyMetrics } from '../../mic-engine';

export interface EntropyMetrics {
  shannonH: number;
  channelCapacity: number;
  fisherInfo: number;
  crlbPixels: number;
  snr: number;
}

export function computeEntropy(
  image: Float32Array,
  width: number,
  height: number,
): EntropyMetrics {
  const r = computeEntropyMetrics({ data: image, width, height });
  return {
    shannonH: r.shannonEntropy,
    channelCapacity: r.channelCapacity,
    fisherInfo: r.fisherInformation,
    crlbPixels: r.crlbPixels,
    snr: r.snr,
  };
}
