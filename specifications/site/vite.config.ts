import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site reads the specifications one directory up (../architecture,
// ../specs, ../registry) at build time, so the Markdown and JSON stay the
// single source of truth. `fs.allow` lets the dev server serve them.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { fs: { allow: [".."] } },
  build: { chunkSizeWarningLimit: 3000 },
});
