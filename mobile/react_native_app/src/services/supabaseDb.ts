/**
 * supabaseDb.ts — Database helpers for screenings, analytics, sleep & feeding logs
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

/**
 * Fetch ALL screening records across all users (admin only).
 * Requires the "Admins view all screenings" RLS policy on the screenings table.
 */
export async function getAllScreenings(limit = 500): Promise<ScreeningRow[]> {
  const { data, error } = await supabase
    .from('screenings')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch all screenings: ${error.message}`);
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

// ── Sleep Logs ───────────────────────────────────────────────────────────────

export interface SleepLogRow {
  id: string;
  user_id: string;
  start_time: string;
  end_time: string;
  duration_hours: number;
  notes: string;
  date: string;
  created_at: string;
}

/**
 * Insert a sleep log entry.
 */
export async function saveSleepLog(
  record: Omit<SleepLogRow, 'id' | 'created_at'>,
): Promise<void> {
  const { error } = await supabase.from('sleep_logs').insert(record);
  if (error) {
    console.warn('Failed to save sleep log:', error.message);
  }
}

/**
 * Fetch the authenticated user's sleep logs.
 */
export async function getSleepLogs(
  userId: string,
  limit = 200,
): Promise<SleepLogRow[]> {
  const { data, error } = await supabase
    .from('sleep_logs')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch sleep logs: ${error.message}`);
  return (data ?? []) as SleepLogRow[];
}

// ── Feeding Logs ─────────────────────────────────────────────────────────────

export interface FeedingLogRow {
  id: string;
  user_id: string;
  type: 'breastfeeding' | 'formula' | 'solid';
  time: string;
  quantity_ml?: number;
  notes: string;
  date: string;
  created_at: string;
}

/**
 * Insert a feeding log entry.
 */
export async function saveFeedingLog(
  record: Omit<FeedingLogRow, 'id' | 'created_at'>,
): Promise<void> {
  const { error } = await supabase.from('feeding_logs').insert(record);
  if (error) {
    console.warn('Failed to save feeding log:', error.message);
  }
}

/**
 * Fetch the authenticated user's feeding logs.
 */
export async function getFeedingLogs(
  userId: string,
  limit = 200,
): Promise<FeedingLogRow[]> {
  const { data, error } = await supabase
    .from('feeding_logs')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch feeding logs: ${error.message}`);
  return (data ?? []) as FeedingLogRow[];
}
