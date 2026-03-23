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
const SUPABASE_URL = 'https://clmvkxgmpytgztthrdmk.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNsbXZreGdtcHl0Z3p0dGhyZG1rIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxNjI0OTUsImV4cCI6MjA4OTczODQ5NX0.1yr3LnYUnqlh6yahH0a7odjaPugRkiXMPZ_81yp4fiw';

/** True only when real credentials have been configured. */
export const isSupabaseConfigured =
  !SUPABASE_URL.includes('YOUR_PROJECT_REF') &&
  !SUPABASE_ANON_KEY.includes('YOUR_ANON_KEY');

// ── Secure-store adapter (native only) ───────────────────────────────────────
const ExpoSecureStoreAdapter = {
  getItem: (key: string) => SecureStore.getItemAsync(key),
  setItem: (key: string, value: string) => SecureStore.setItemAsync(key, value),
  removeItem: (key: string) => SecureStore.deleteItemAsync(key),
};

// ── Create the Supabase client ───────────────────────────────────────────────
// When credentials are not configured:
//   - skipAutoInitialize: true  → prevents the SDK from calling _initialize()
//     in the constructor, which would trigger _recoverAndRefresh() and make
//     a network request to the placeholder URL on every app start.
//   - autoRefreshToken / persistSession: false → no background refresh or storage.
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    // Use secure storage on native; on web Supabase SDK uses localStorage
    ...(Platform.OS !== 'web' && isSupabaseConfigured ? { storage: ExpoSecureStoreAdapter } : {}),
    autoRefreshToken: isSupabaseConfigured,
    persistSession: isSupabaseConfigured,
    skipAutoInitialize: !isSupabaseConfigured,
    detectSessionInUrl: false, // not needed in React Native
  },
});
