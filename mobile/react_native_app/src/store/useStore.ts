/**
 * useStore.ts — Zustand state management
 *
 * Expanded with Supabase persistence for screening history,
 * user auth state, offline support, role-based routing,
 * and child health tracking (sleep, feeding).
 */
import { Platform } from 'react-native';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import * as SecureStore from 'expo-secure-store';
import type { PredictionResponse } from '../services/types';
import type { QueuedRequest } from '../services/offlineQueue';
import { saveScreeningResult, getScreeningHistory, saveSleepLog, saveFeedingLog } from '../services/supabaseDb';
import type { Language } from '../i18n/translations';
import type { UserRole } from '../services/supabaseAuth';

// ── Cross-platform storage adapter ───────────────────────────────────────────
// SecureStore is native-only; fall back to localStorage on web.
const secureStorage = createJSONStorage(() =>
  Platform.OS === 'web'
    ? {
        getItem: (key: string) => Promise.resolve(localStorage.getItem(key)),
        setItem: (key: string, value: string) => {
          localStorage.setItem(key, value);
          return Promise.resolve();
        },
        removeItem: (key: string) => {
          localStorage.removeItem(key);
          return Promise.resolve();
        },
      }
    : {
        getItem: (key: string) => SecureStore.getItemAsync(key),
        setItem: (key: string, value: string) => SecureStore.setItemAsync(key, value),
        removeItem: (key: string) => SecureStore.deleteItemAsync(key),
      },
);

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

// ── Sleep Log ──
export interface SleepLog {
  id: string;
  startTime: string;
  endTime: string;
  durationHours: number;
  notes: string;
  date: string;
}

// ── Feeding Log ──
export interface FeedingLog {
  id: string;
  type: 'breastfeeding' | 'formula' | 'solid';
  time: string;
  quantityMl?: number;
  notes: string;
  date: string;
}

interface AppState {
  // ── App settings ──
  language: Language;
  hasSeenOnboarding: boolean;
  setLanguage: (lang: Language) => void;
  setHasSeenOnboarding: (seen: boolean) => void;

  // ── Auth ──
  userId: string | null;
  role: UserRole | null;

  // ── Child profile ──
  childName: string | null;
  childAgeMonths: number | null;
  childGender: 'male' | 'female' | null;

  // ── Data ──
  result: PredictionResponse | null;
  imageUri: string | null;

  // ── History ──
  history: ScreeningRecord[];

  // ── Child health logs ──
  sleepLogs: SleepLog[];
  feedingLogs: FeedingLog[];

  // ── Offline ──
  offlineQueue: QueuedRequest[];
  lastSyncedAt: Date | null;

  // ── Education progress ──
  completedItems: string[];

  // ── UI State ──
  isLoading: boolean;
  error: string | null;

  // ── Actions ──
  setUserId: (id: string | null) => void;
  setRole: (role: UserRole | null) => void;
  setChildProfile: (name: string | null, ageMonths: number | null, gender: 'male' | 'female' | null) => void;
  setResult: (data: PredictionResponse) => void;
  setImageUri: (uri: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addToHistory: (record: ScreeningRecord) => void;
  loadHistoryFromSupabase: (userId: string) => Promise<void>;
  clearHistory: () => void;
  addSleepLog: (log: SleepLog) => void;
  addFeedingLog: (log: FeedingLog) => void;
  addToQueue: (request: QueuedRequest) => void;
  removeFromQueue: (id: string) => void;
  setLastSyncedAt: (date: Date) => void;
  toggleCompleted: (itemId: string) => void;
  clearCompleted: () => void;
  reset: () => void;
}

const initialState = {
  language: 'en' as Language,
  hasSeenOnboarding: false,
  userId: null as string | null,
  role: null as UserRole | null,
  childName: null as string | null,
  childAgeMonths: null as number | null,
  childGender: null as 'male' | 'female' | null,
  result: null,
  imageUri: null,
  history: [] as ScreeningRecord[],
  sleepLogs: [] as SleepLog[],
  feedingLogs: [] as FeedingLog[],
  offlineQueue: [] as QueuedRequest[],
  lastSyncedAt: null as Date | null,
  completedItems: [] as string[],
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
  setRole: (role) => set({ role }),

  setChildProfile: (name, ageMonths, gender) =>
    set({ childName: name, childAgeMonths: ageMonths, childGender: gender }),

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

  addSleepLog: (log) => {
    set((state) => ({
      sleepLogs: [log, ...state.sleepLogs].slice(0, 200),
    }));
    // Sync to Supabase (fire-and-forget)
    const userId = get().userId;
    if (userId) {
      saveSleepLog({
        user_id: userId,
        start_time: log.startTime,
        end_time: log.endTime,
        duration_hours: log.durationHours,
        notes: log.notes,
        date: log.date,
      }).catch((err) => {
        console.warn('Failed to persist sleep log to Supabase:', err.message);
      });
    }
  },

  addFeedingLog: (log) => {
    set((state) => ({
      feedingLogs: [log, ...state.feedingLogs].slice(0, 200),
    }));
    // Sync to Supabase (fire-and-forget)
    const userId = get().userId;
    if (userId) {
      saveFeedingLog({
        user_id: userId,
        type: log.type,
        time: log.time,
        quantity_ml: log.quantityMl,
        notes: log.notes,
        date: log.date,
      }).catch((err) => {
        console.warn('Failed to persist feeding log to Supabase:', err.message);
      });
    }
  },

  addToQueue: (request) =>
    set((state) => ({
      offlineQueue: [...state.offlineQueue, request],
    })),
  removeFromQueue: (id) =>
    set((state) => ({
      offlineQueue: state.offlineQueue.filter((r) => r.id !== id),
    })),
  setLastSyncedAt: (date) => set({ lastSyncedAt: date }),

  toggleCompleted: (itemId) => set((state) => {
    const idx = state.completedItems.indexOf(itemId);
    return {
      completedItems: idx >= 0
        ? state.completedItems.filter((id) => id !== itemId)
        : [...state.completedItems, itemId],
    };
  }),

  clearCompleted: () => set({ completedItems: [] }),

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
      partialize: (state) => ({
        language: state.language,
        hasSeenOnboarding: state.hasSeenOnboarding,
        role: state.role,
        childName: state.childName,
        childAgeMonths: state.childAgeMonths,
        childGender: state.childGender,
        // Persist so past results are viewable offline and queue survives restarts
        history: state.history,
        offlineQueue: state.offlineQueue,
        lastSyncedAt: state.lastSyncedAt,
        completedItems: state.completedItems,
      }),
    },
  ),
);
