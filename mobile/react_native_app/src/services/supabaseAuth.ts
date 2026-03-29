/**
 * supabaseAuth.ts — Authentication helpers
 *
 * Wraps Supabase Auth methods for sign-up, sign-in, sign-out,
 * session listening, current-user retrieval, and role-based profiles.
 */
import type { AuthChangeEvent, Session, User } from '@supabase/supabase-js';
import { supabase } from './supabase';

export type UserRole = 'admin' | 'chw' | 'parent';

export interface UserProfile {
  id: string;
  role: UserRole;
  full_name: string | null;
}

// ── Sign Up ──────────────────────────────────────────────────────────────────
export async function signUp(email: string, password: string) {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: { emailRedirectTo: 'https://clmvkxgmpytgztthrdmk.supabase.co' },
  });
  if (error) throw new Error(error.message);
  if (data.user && data.user.identities && data.user.identities.length === 0) {
    throw new Error('An account with this email already exists. Please sign in.');
  }
  return data;
}

// ── Sign Up with Role ─────────────────────────────────────────────────────────
export async function signUpWithRole(
  email: string,
  password: string,
  role: UserRole,
  fullName?: string,
) {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: 'https://clmvkxgmpytgztthrdmk.supabase.co',
      data: { role, full_name: fullName ?? '' },
    },
  });
  if (error) throw new Error(error.message);
  if (data.user && data.user.identities && data.user.identities.length === 0) {
    throw new Error('An account with this email already exists. Please sign in.');
  }
  return data;
}

// ── Password Reset ───────────────────────────────────────────────────────────
export async function sendPasswordReset(email: string) {
  // Use the Supabase project URL as redirect — the user resets on the
  // hosted page, then returns to the app and signs in with the new password.
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: 'https://clmvkxgmpytgztthrdmk.supabase.co',
  });
  if (error) throw new Error(error.message);
}

export async function updatePassword(newPassword: string) {
  const { error } = await supabase.auth.updateUser({ password: newPassword });
  if (error) throw new Error(error.message);
}

// ── Sign In ──────────────────────────────────────────────────────────────────
export async function signIn(email: string, password: string) {
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
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
  const { data: { user } } = await supabase.auth.getUser();
  return user;
}

// ── Get Current Session ──────────────────────────────────────────────────────
export async function getSession(): Promise<Session | null> {
  const { data: { session } } = await supabase.auth.getSession();
  return session;
}

// ── Get User Profile ─────────────────────────────────────────────────────────
export async function getUserProfile(): Promise<UserProfile | null> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;
  const { data, error } = await supabase
    .from('profiles')
    .select('id, role, full_name')
    .eq('id', user.id)
    .single();
  if (error || !data) {
    // Fallback: read role from user metadata
    const role = (user.user_metadata?.role as UserRole) ?? 'parent';
    return { id: user.id, role, full_name: user.user_metadata?.full_name ?? null };
  }
  return data as UserProfile;
}

// ── Auth State Listener ──────────────────────────────────────────────────────
/**
 * Subscribe to auth state changes (sign-in, sign-out, token refresh, etc.).
 * Returns an unsubscribe function.
 */
export function onAuthStateChanged(
  callback: (event: AuthChangeEvent, session: Session | null) => void,
) {
  const { data: { subscription } } = supabase.auth.onAuthStateChange(callback);
  return () => subscription.unsubscribe();
}
