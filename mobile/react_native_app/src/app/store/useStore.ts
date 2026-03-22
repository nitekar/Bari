/**
 * useStore.ts — Zustand state management
 *
 * Expanded with Supabase persistence for screening history,
 * user auth state, and offline support.
 */
import { create } from 'zustand';
import type { PredictionResponse } from '../../services/types';
import type { QueuedRequest } from '../../services/offlineQueue';
import { saveScreeningResult, getScreeningHistory } from '../../services/supabaseDb';

// ── Screening History Record ──
export interface ScreeningRecord {
  id: string;
  date: string; // ISO 8601
  prediction: string;
  confidence: number;
  mode: 'tabular' | 'image' | 'multimodal';
  age?: number;
  gender?: number;
  imageUrl?: string | null;
}

interface AppState {
  // ── Auth ──
  userId: string | null;

  // ── Data ──
  result: PredictionResponse | null;
  imageUri: string | null;

  // ── History ──
  history: ScreeningRecord[];

  // ── Offline ──
  offlineQueue: QueuedRequest[];
  lastSyncedAt: Date | null;

  // ── UI State ──
  isLoading: boolean;
  error: string | null;

  // ── Actions ──
  setUserId: (id: string | null) => void;
  setResult: (data: PredictionResponse) => void;
  setImageUri: (uri: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addToHistory: (record: ScreeningRecord) => void;
  loadHistoryFromSupabase: (userId: string) => Promise<void>;
  clearHistory: () => void;
  addToQueue: (request: QueuedRequest) => void;
  removeFromQueue: (id: string) => void;
  setLastSyncedAt: (date: Date) => void;
  reset: () => void;
}

const initialState = {
  userId: null as string | null,
  result: null,
  imageUri: null,
  history: [] as ScreeningRecord[],
  offlineQueue: [] as QueuedRequest[],
  lastSyncedAt: null as Date | null,
  isLoading: false,
  error: null,
};

export const useStore = create<AppState>((set, get) => ({
  ...initialState,

  setUserId: (id) => set({ userId: id }),

  setResult: (data) => set({ result: data, error: null }),
  setImageUri: (uri) => set({ imageUri: uri }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error, isLoading: false }),

  addToHistory: (record) => {
    // Optimistically add to local state
    set((state) => ({
      history: [record, ...state.history].slice(0, 100),
    }));

    // Persist to Supabase (fire-and-forget)
    const userId = get().userId;
    if (userId) {
      saveScreeningResult({
        user_id: userId,
        prediction: record.prediction,
        confidence: record.confidence,
        mode: record.mode,
        age: record.age,
        gender: record.gender,
        image_url: record.imageUrl ?? null,
      }).catch((err) => {
        console.warn('Failed to persist screening to Supabase:', err.message);
      });
    }
  },

  loadHistoryFromSupabase: async (userId) => {
    try {
      const rows = await getScreeningHistory(userId);
      const records: ScreeningRecord[] = rows.map((row) => ({
        id: row.id,
        date: row.created_at,
        prediction: row.prediction,
        confidence: row.confidence,
        mode: row.mode,
        age: row.age,
        gender: row.gender,
        imageUrl: row.image_url,
      }));
      set({ history: records });
    } catch (err: any) {
      console.warn('Failed to load history from Supabase:', err.message);
    }
  },

  clearHistory: () => set({ history: [] }),

  addToQueue: (request) =>
    set((state) => ({
      offlineQueue: [...state.offlineQueue, request],
    })),
  removeFromQueue: (id) =>
    set((state) => ({
      offlineQueue: state.offlineQueue.filter((r) => r.id !== id),
    })),
  setLastSyncedAt: (date) => set({ lastSyncedAt: date }),

  reset: () =>
    set({
      result: null,
      imageUri: null,
      isLoading: false,
      error: null,
    }),
}));
