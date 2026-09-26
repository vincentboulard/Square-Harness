import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// Every browser that runs this interface reads WOFF2; do not ship KaTeX's
// WOFF and TTF duplicates inside the Python package.
function katexWoff2Only(): Plugin {
  return {
    name: 'katex-woff2-only',
    enforce: 'pre',
    transform(code, id) {
      if (!/katex(\.min)?\.css$/.test(id)) return null
      return code
        .replace(/,\s*url\([^)]*\.woff\)\s*format\(["']woff["']\)/g, '')
        .replace(/,\s*url\([^)]*\.ttf\)\s*format\(["']truetype["']\)/g, '')
    },
  }
}

export default defineConfig({
  plugins: [react(), katexWoff2Only()],
  base: '/',
  build: {
    outDir: '../mathagent/gui/static',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    // `npm run dev` talks to a running `square-harness --gui` (default port 8765).
    proxy: { '/api': { target: 'http://127.0.0.1:8765', changeOrigin: false } },
  },
})
