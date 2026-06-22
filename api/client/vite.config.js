import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// The FastAPI app serves index.html from api/static and mounts assets at /static.
// So we build into ../static and set the public base path to /static/.
export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: fileURLToPath(new URL("../static", import.meta.url)),
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` -> proxy API calls to the FastAPI backend.
    proxy: {
      "/stt": "http://localhost:8000",
      "/check": "http://localhost:8000",
    },
  },
});
