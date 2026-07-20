// Scale field estimation — thin wrapper over mic-engine Algorithm 1
// Returns α(x,y) in µm/px, power-law exponent, and chart-ready spectral data

import { estimateScaleField } from '../../mic-engine';

export interface ScaleField {
  alpha: Float32Array;       // α(x,y), µm/px per pixel
  mean: number;              // ᾱ
  stddev: number;            // σ_α
  powerLawExponent: number;  // slope of log-log spectral decay
  width: number;
  height: number;
}

export interface SpectralPowerPoint { freq: number; energy: number; }
export interface ScaleHistogramBin  { alpha: number; count: number; }

export interface ScaleFieldOutput {
  field: ScaleField;
  spectralPower: SpectralPowerPoint[];
  scaleHistogram: ScaleHistogramBin[];
  channelCapacity: number;
  snr: number;
}

export function computeScaleField(
  image: Float32Array,
  width: number,
  height: number,
): ScaleFieldOutput {
  const result = estimateScaleField({ data: image, width, height });
  const alpha = result.alpha;

  // mean and stddev
  let sum = 0;
  for (let i = 0; i < alpha.length; i++) sum += alpha[i];
  const mean = sum / alpha.length;
  let varSum = 0;
  for (let i = 0; i < alpha.length; i++) varSum += (alpha[i] - mean) ** 2;
  const stddev = Math.sqrt(varSum / alpha.length);

  // Build spectral power series for D3 chart (radial average, log-spaced)
  // We approximate from the power-law: |û(k)|² ∝ k^exponent
  const spectralPower: SpectralPowerPoint[] = [];
  const exp = result.powerLawExponent;
  for (let i = 0; i < 32; i++) {
    const freq = Math.pow(10, -2 + i * (2 / 31));
    const energy = Math.pow(freq, exp);
    spectralPower.push({ freq, energy });
  }

  // Scale histogram: bin α values into 20 bins
  const scaleHistogram: ScaleHistogramBin[] = [];
  let minA = Infinity, maxA = -Infinity;
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] < minA) minA = alpha[i];
    if (alpha[i] > maxA) maxA = alpha[i];
  }
  const nBins = 20;
  const binWidth = (maxA - minA) / nBins || 0.001;
  const counts = new Array(nBins).fill(0);
  for (let i = 0; i < alpha.length; i++) {
    const bin = Math.min(nBins - 1, Math.floor((alpha[i] - minA) / binWidth));
    counts[bin]++;
  }
  for (let b = 0; b < nBins; b++) {
    scaleHistogram.push({ alpha: minA + (b + 0.5) * binWidth, count: counts[b] });
  }

  // SNR and channel capacity from image statistics
  let sigSum = 0, noiseSum = 0, n = 0;
  const threshold = 0.5;
  for (let i = 0; i < image.length; i++) {
    if (image[i] > threshold) { sigSum += image[i]; n++; }
    else noiseSum += image[i];
  }
  const bg = noiseSum / (image.length - n + 1);
  const signal = n > 0 ? sigSum / n : 1;
  const snr = bg > 0 ? signal / bg : 10;
  const channelCapacity = 0.5 * Math.log2(1 + snr);

  return {
    field: { alpha, mean, stddev, powerLawExponent: exp, width, height },
    spectralPower,
    scaleHistogram,
    channelCapacity,
    snr,
  };
}
