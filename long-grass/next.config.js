/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
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
    };
    return config;
  },
};

module.exports = nextConfig;
