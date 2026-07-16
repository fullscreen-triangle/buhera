/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // @sachikonye/sbs is an ESM package linked in via `npm link` (its source of
  // truth lives in the hegel repo at consequences/src/lib/sbs). Next does not
  // transpile node_modules by default, and a linked package resolves there;
  // listing it here makes Next compile it like local source, so its ESM and
  // browser globals (document, WebGL2) work in the client bundle.
  transpilePackages: ["@sachikonye/sbs", "scope-lang", "@lavoisier/shapeshifter"],
  webpack: (config) => {
    // @xenova/transformers is dynamically imported by the turbulance
    // model resolvers (research / ask primitives). The package is heavy
    // and not needed for the deterministic core. Mark it as a fallback
    // so the build does not fail when the package is not installed;
    // the dynamic import will reject at runtime if turbulance tries
    // to use a model primitive without the package present.
    config.resolve.fallback = {
      ...(config.resolve.fallback || {}),
      "@xenova/transformers": false,
      // purpose's llm.js imports the Anthropic SDK unconditionally, but it is
      // only invoked when LLM_PROVIDER=anthropic. Marking it as a fallback
      // keeps the build green for the HuggingFace-only path (recommended).
      // Install @anthropic-ai/sdk to unlock the Anthropic provider.
      "@anthropic-ai/sdk": false,
    };
    return config;
  },
};

module.exports = nextConfig;
