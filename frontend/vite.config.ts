import { defineConfig } from "vite";
export default defineConfig({
  server: { strictPort: true },
  build: { rollupOptions: { input: "brain.html" } },
});
