/**
 * supabaseDb.ts — Database helpers for screenings & analytics events
 *
 * All queries are scoped by user_id via RLS policies on the Supabase side,
 * but we also pass user_id explicitly for clarity and type safety.
 */
import { supabase } from './supabase';
import type { ScreeningRow, AnalyticsEventRow } from './types';

// ── Screenings ───────────────────────────────────────────────────────────────

/**
 * Insert a new screening result.
 */
export async function saveScreeningResult(
  record: Omit<ScreeningRow, 'id' | 'created_at'>,
): Promise<ScreeningRow> {
  const { data, error } = await supabase
    .from('screenings')
    .insert(record)
    .select()
    .single();

  if (error) throw new Error(`Failed to save screening: ${error.message}`);
  return data as ScreeningRow;
}

/**
 * Fetch the authenticated user's screening history (newest first).
 */
export async function getScreeningHistory(
  userId: string,
  limit = 100,
): Promise<ScreeningRow[]> {
  const { data, error } = await supabase
    .from('screenings')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch history: ${error.message}`);
  return (data ?? []) as ScreeningRow[];
}

// ── Analytics Events ─────────────────────────────────────────────────────────

/**
 * Insert an analytics event.
 */
export async function saveAnalyticsEvent(
  record: Omit<AnalyticsEventRow, 'id' | 'created_at'>,
): Promise<void> {
  const { error } = await supabase.from('analytics_events').insert(record);
  if (error) {
    // Fire-and-forget — log but don't throw
    console.warn('Failed to save analytics event:', error.message);
  }
}

/**
 * Fetch the authenticated user's analytics events.
 */
export async function getAnalyticsEvents(
  userId: string,
  limit = 500,
): Promise<AnalyticsEventRow[]> {
  const { data, error } = await supabase
    .from('analytics_events')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch events: ${error.message}`);
  return (data ?? []) as AnalyticsEventRow[];
}
