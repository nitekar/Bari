/**
 * useStore.ts — Zustand state management
 *
 * Expanded with Supabase persistence for screening history,
 * user auth state, and offline support.
 */
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import * as SecureStore from 'expo-secure-store';
import type { PredictionResponse } from '../services/types';
import type { QueuedRequest } from '../services/offlineQueue';
import { saveScreeningResult, getScreeningHistory } from '../services/supabaseDb';
import type { Language } from '../i18n/translations';

// ── SecureStore adapter for zustand/persist ──────────────────────────────────
const secureStorage = createJSONStorage(() => ({
  getItem: (key: string) => SecureStore.getItemAsync(key),
  setItem: (key: string, value: string) => SecureStore.setItemAsync(key, value),
  removeItem: (key: string) => SecureStore.deleteItemAsync(key),
}));

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
  patientName?: string;
  patientLocation?: string;
}

interface AppState {
  // ── App settings ──
  language: Language;
  hasSeenOnboarding: boolean;
  setLanguage: (lang: Language) => void;
  setHasSeenOnboarding: (seen: boolean) => void;

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
  language: 'en' as Language,
  hasSeenOnboarding: false,
  userId: null as string | null,
  result: null,
  imageUri: null,
  history: [] as ScreeningRecord[],
  offlineQueue: [] as QueuedRequest[],
  lastSyncedAt: null as Date | null,
  isLoading: false,
  error: null,
};

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
  ...initialState,

  setLanguage: (lang) => set({ language: lang }),
  setHasSeenOnboarding: (seen) => set({ hasSeenOnboarding: seen }),

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
        patient_name: record.patientName ?? null,
        patient_location: record.patientLocation ?? null,
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
    }),
    {
      name: 'bari-app-storage',
      storage: secureStorage,
      // Only persist user preferences — not ephemeral runtime state
      partialize: (state) => ({
        language: state.language,
        hasSeenOnboarding: state.hasSeenOnboarding,
      }),
    },
  ),
);
