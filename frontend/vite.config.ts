import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API requests to the backend server
      '/api': {
        target: 'http://127.0.0.1:8000', // Your backend address
        changeOrigin: true,
        // Add rewrite rule to remove '/api' prefix
        rewrite: (path) => path.replace(/^\/api/, ''), 
      },
      // Add other proxies if needed, e.g., for WebSockets if facing issues later
      // '/ws': {
      //   target: 'ws://127.0.0.1:8000',
      //   ws: true,
      // }
    }
  }
})
