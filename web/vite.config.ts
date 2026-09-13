import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API stays on FastAPI (the models, the feature pipeline and the parquet
// panel are all Python), so in development Vite proxies the API routes to
// uvicorn. In production `npm run build` writes web/dist, which serve/api.py
// mounts at "/" -- one origin, no proxy, no CORS.
const API = "http://127.0.0.1:8000";
const API_ROUTES = [
  "/health", "/models", "/metrics", "/dates", "/facets",
  "/search", "/signals", "/categories", "/picks", "/tickers", "/predict",
  "/saudi",
];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [route, { target: API, changeOrigin: true }]),
    ),
  },
  build: { outDir: "dist", sourcemap: true },
});
