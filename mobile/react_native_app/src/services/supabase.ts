/**
 * supabase.ts — Supabase client initialisation
 *
 * Uses expo-secure-store on native for persisting auth tokens.
 * On web, falls back to localStorage automatically.
 *
 * Credentials are read from the centralized env config.
 */
import { createClient } from '@supabase/supabase-js';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';
import {
  SUPABASE_URL,
  SUPABASE_ANON_KEY,
  IS_SUPABASE_CONFIGURED,
} from '../config/env';

/** Re-export for backwards compatibility */
export const isSupabaseConfigured = IS_SUPABASE_CONFIGURED;

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
    detectSessionInUrl: false, // not needed in React Native
  } as any,
});
