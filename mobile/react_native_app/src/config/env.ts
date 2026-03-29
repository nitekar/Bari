/**
 * env.ts — Centralized environment configuration
 *
 * Reads values from Expo Constants (app.config.js → extra).
 * Fallbacks are intentionally placeholder values so the app
 * gracefully degrades when credentials aren't configured.
 */
import Constants from 'expo-constants';

const extra = Constants.expoConfig?.extra ?? {};

/** Supabase project URL */
export const SUPABASE_URL: string =
  extra.supabaseUrl ?? 'https://YOUR_PROJECT_REF.supabase.co';

/** Supabase anonymous key (public, safe for client usage behind RLS) */
export const SUPABASE_ANON_KEY: string =
  extra.supabaseAnonKey ?? 'YOUR_ANON_KEY';

/** FastAPI backend base URL */
export const API_BASE_URL: string =
  extra.apiBaseUrl ?? 'https://YOUR_BACKEND_URL';

/** API key for authenticating requests to the FastAPI backend */
export const API_KEY: string =
  extra.apiKey ?? '';

/** True when real Supabase credentials are present */
export const IS_SUPABASE_CONFIGURED =
  !!SUPABASE_URL &&
  !SUPABASE_URL.includes('YOUR_PROJECT_REF') &&
  !!SUPABASE_ANON_KEY &&
  !SUPABASE_ANON_KEY.includes('YOUR_ANON_KEY');
