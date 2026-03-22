/**
 * supabase.ts — Supabase client initialisation
 *
 * Uses expo-secure-store on native for persisting auth tokens.
 * On web, falls back to localStorage automatically.
 *
 * ⚠️  Replace SUPABASE_URL and SUPABASE_ANON_KEY with your project values.
 */
import { createClient } from '@supabase/supabase-js';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

// ── Your Supabase project credentials ────────────────────────────────────────
const SUPABASE_URL = 'https://YOUR_PROJECT_REF.supabase.co';
const SUPABASE_ANON_KEY = 'YOUR_ANON_KEY';

// ── Secure-store adapter (native only) ───────────────────────────────────────
const ExpoSecureStoreAdapter = {
  getItem: (key: string) => SecureStore.getItemAsync(key),
  setItem: (key: string, value: string) => SecureStore.setItemAsync(key, value),
  removeItem: (key: string) => SecureStore.deleteItemAsync(key),
};

// ── Create the Supabase client ───────────────────────────────────────────────
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    // Use secure storage on native; on web Supabase SDK uses localStorage
    ...(Platform.OS !== 'web' ? { storage: ExpoSecureStoreAdapter } : {}),
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false, // not needed in React Native
  },
});
