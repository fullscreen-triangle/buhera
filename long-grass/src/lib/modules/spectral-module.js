// spectral — gospel vivid-symbolism's numerical core (vendor/spectral,
// byte-exact): spectral embeddings, shader-kernel cosine ranking, FFT
// matched-filter motif scanning. Specification: specs/spectral.md.
import * as embedding from "@gospel/spectral/embedding";
import * as matchedFilter from "@gospel/spectral/matched_filter";
import { makeSpectralModule } from "@buhera/registry/modules";

export const spectralEngine = { ...embedding, ...matchedFilter };
export const spectralModule = makeSpectralModule(spectralEngine);
