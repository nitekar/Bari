/**
 * api.ts — Axios instance with interceptors and optional mock adapter
 */
import axios from 'axios';
import { attachMockAdapter } from './mockAdapter';

/**
 * ⚠️  Replace with your actual backend URL before deploying.
 * For local development use your machine's IP (not localhost)
 * so the Expo Go app on a physical device can reach it.
 *
 * Examples:
 *   http://192.168.1.100:8000
 *   https://your-api.railway.app
 */
const BASE_URL = 'https://web-production-c7c1.up.railway.app';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000, // 30 s — image uploads can be slow
  headers: {
    Accept: 'application/json',
  },
});

// ── Request interceptor ──────────────────────────────────────────────────────
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here in the future
    return config;
  },
  (error) => Promise.reject(error),
);

// ── Response interceptor ─────────────────────────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with an error status
      const detail =
        error.response.data?.detail || 'Something went wrong on the server.';
      return Promise.reject(new Error(detail));
    }
    if (error.request) {
      // No response received
      return Promise.reject(
        new Error('Unable to reach the server. Please check your connection.'),
      );
    }
    return Promise.reject(error);
  },
);

// ── Attach mock adapter only when backend URL is not configured ───────────────
if (BASE_URL.includes('YOUR_BACKEND_URL')) {
  attachMockAdapter(api);
}

export default api;
