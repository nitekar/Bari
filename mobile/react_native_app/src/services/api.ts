/**
 * api.ts — Axios instance with auth, interceptors, and optional mock adapter
 *
 * Sends X-API-Key header on every request for backend authentication.
 */
import axios from 'axios';
import { attachMockAdapter } from './mockAdapter';
import { API_BASE_URL, API_KEY } from '../config/env';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000, // 30 s — image uploads can be slow
  headers: {
    Accept: 'application/json',
  },
  maxContentLength: 10 * 1024 * 1024, // 10 MB max response
  maxBodyLength: 10 * 1024 * 1024,    // 10 MB max request body
});

// ── Request interceptor ──────────────────────────────────────────────────────
api.interceptors.request.use(
  (config) => {
    // Attach API key to every request
    if (API_KEY) {
      config.headers['X-API-Key'] = API_KEY;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// ── Response interceptor ─────────────────────────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const status = error.response.status;
      if (status === 401 || status === 403) {
        return Promise.reject(new Error('Authentication failed. Please check API credentials.'));
      }
      if (status === 429) {
        return Promise.reject(new Error('Too many requests. Please wait a moment and try again.'));
      }
      const detail =
        error.response.data?.detail || 'Something went wrong on the server.';
      return Promise.reject(new Error(detail));
    }
    if (error.request) {
      return Promise.reject(
        new Error('Unable to reach the server. Please check your connection.'),
      );
    }
    return Promise.reject(error);
  },
);

// ── Attach mock adapter only when backend URL is not configured ───────────────
if (API_BASE_URL.includes('YOUR_BACKEND_URL')) {
  attachMockAdapter(api);
}

export default api;
