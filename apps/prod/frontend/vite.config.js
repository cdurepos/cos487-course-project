import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Proxies /search to the FastAPI backend so the browser can call a same-origin
 * route. Point elsewhere with SEARCH_BACKEND_URL if needed.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/search': {
        target: process.env.SEARCH_BACKEND_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
