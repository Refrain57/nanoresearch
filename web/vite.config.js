import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'
import http from 'node:http'

export default defineConfig({
  plugins: [
    vue(),
    {
      name: 'file-direct-proxy',
      configureServer(server) {
        server.middlewares.stack.unshift({ route: '', handle: (req, res, next) => {
          if (!req.url?.includes('/documents/') || !req.url?.includes('/file')) {
            return next();
          }
          const headers = { ...req.headers, host: 'localhost:8000' }
          delete headers.range
          delete headers['if-range']
          headers['cache-control'] = 'no-cache'
          const proxyReq = http.request({
            hostname: 'localhost',
            port: 8000,
            path: req.url,
            method: req.method,
            headers,
          }, (proxyRes) => {
            proxyRes.headers['content-type'] = 'application/octet-stream'
            proxyRes.headers['cache-control'] = 'no-store'
            delete proxyRes.headers['transfer-encoding']
            delete proxyRes.headers['content-encoding']
            if (proxyRes.headers['content-disposition']) {
              proxyRes.headers['content-disposition'] =
                proxyRes.headers['content-disposition'].replace(/\.pdf"/gi, '"')
            }
            res.writeHead(proxyRes.statusCode, proxyRes.headers)
            let total = 0
            proxyRes.on('data', (c) => { total += c.length })
            proxyRes.on('end', () => { console.log('[file-proxy] OK:', total, 'bytes') })
            proxyRes.on('error', (e) => { console.error('[file-proxy] STREAM ERR:', e.message) })
            res.on('close', () => { console.log('[file-proxy] RES closed, sent:', total, 'bytes') })
            proxyRes.pipe(res)
          })
          proxyReq.on('error', (e) => {
            console.error('[file-proxy] REQ ERR:', e.message)
            if (!res.headersSent) { res.statusCode = 502; res.end() }
          })
          req.pipe(proxyReq)
        }})
      },
    },
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req, res) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.on('data', () => res.flush?.())
            }
          })
        },
      },
    },
  },
})
