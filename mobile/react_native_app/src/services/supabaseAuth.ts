/**
 * supabaseAuth.ts — Authentication helpers
 *
 * Wraps Supabase Auth methods for sign-up, sign-in, sign-out,
 * session listening, and current-user retrieval.
 */
import type { AuthChangeEvent, Session, User } from '@supabase/supabase-js';
import { supabase } from './supabase';

// ── Sign Up ──────────────────────────────────────────────────────────────────
export async function signUp(email: string, password: string) {
  const { data, error } = await supabase.auth.signUp({ email, password });
  if (error) throw new Error(error.message);
  return data;
}

// ── Sign In ──────────────────────────────────────────────────────────────────
export async function signIn(email: string, password: string) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw new Error(error.message);
  return data;
}

// ── Sign Out ─────────────────────────────────────────────────────────────────
export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) throw new Error(error.message);
}

// ── Get Current User ─────────────────────────────────────────────────────────
export async function getCurrentUser(): Promise<User | null> {
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user;
}

// ── Get Current Session ──────────────────────────────────────────────────────
export async function getSession(): Promise<Session | null> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  return session;
}

// ── Auth State Listener ──────────────────────────────────────────────────────
/**
 * Subscribe to auth state changes (sign-in, sign-out, token refresh, etc.).
 * Returns an unsubscribe function.
 */
export function onAuthStateChanged(
  callback: (event: AuthChangeEvent, session: Session | null) => void,
) {
  const {
    data: { subscription },
  } = supabase.auth.onAuthStateChange(callback);
  return () => subscription.unsubscribe();
}
