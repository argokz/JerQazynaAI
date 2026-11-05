import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Адрес API задаётся VITE_API_URL (см. .env.example); прокси не нужен.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
  build: { chunkSizeWarningLimit: 1500 },
});
