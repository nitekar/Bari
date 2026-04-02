/**
 * analyticsStore.ts — Local analytics event tracking with aggregates
 *
 * Tracks user events and provides derived statistics for the dashboard.
 * Events are persisted locally and also synced to Supabase.
 */
import { Platform } from 'react-native';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import * as SecureStore from 'expo-secure-store';
import { saveAnalyticsEvent, getAnalyticsEvents } from '../services/supabaseDb';

// ── Cross-platform storage adapter ───────────────────────────────────────────
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

// ── Event Types ──────────────────────────────────────────────────────────────

export type AnalyticsEventName =
  | 'screening_started'
  | 'screening_completed'
  | 'image_captured'
  | 'result_viewed'
  | 'referral_opened'
  | 'pdf_exported';

export interface AnalyticsEvent {
  id: string;
  name: AnalyticsEventName;
  timestamp: string; // ISO 8601
  metadata?: Record<string, unknown>;
}

export interface AnalyticsAggregates {
  totalScreenings: number;
  severityDistribution: Record<string, number>;
  avgConfidence: number;
  screeningsPerDay: Record<string, number>;
  mostCommonResult: string;
  lastScreeningDate: string | null;
}

// ── Store ────────────────────────────────────────────────────────────────────

interface AnalyticsState {
  events: AnalyticsEvent[];
  userId: string | null;
  setUserId: (id: string | null) => void;
  trackEvent: (name: AnalyticsEventName, metadata?: Record<string, unknown>) => void;
  loadEventsFromSupabase: (userId: string) => Promise<void>;
  getAggregates: () => AnalyticsAggregates;
  clearEvents: () => void;
}

export const useAnalyticsStore = create<AnalyticsState>()(
  persist(
    (set, get) => ({
      events: [],
      userId: null,

      setUserId: (id) => set({ userId: id }),

      trackEvent: (name, metadata) => {
        const event: AnalyticsEvent = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          name,
          timestamp: new Date().toISOString(),
          metadata,
        };

        // Add locally
        set((state) => ({ events: [...state.events, event] }));

        // Sync to Supabase (fire-and-forget)
        const uid = get().userId;
        if (uid) {
          saveAnalyticsEvent({
            user_id: uid,
            event_name: name,
            metadata: metadata ?? {},
          }).catch(() => {
            // Already logged inside saveAnalyticsEvent
          });
        }
      },

      loadEventsFromSupabase: async (userId) => {
        try {
          const rows = await getAnalyticsEvents(userId);
          const events: AnalyticsEvent[] = rows.map((row) => ({
            id: row.id,
            name: row.event_name as AnalyticsEventName,
            timestamp: row.created_at,
            metadata: row.metadata,
          }));
          set({ events });
        } catch (err: any) {
          console.warn('Failed to load analytics events:', err.message);
        }
      },

      getAggregates: () => {
        const { events } = get();

        const completedEvents = events.filter((e) => e.name === 'screening_completed');

        // Severity distribution
        const severityDistribution: Record<string, number> = {};
        let totalConfidence = 0;

        completedEvents.forEach((e) => {
          const severity = (e.metadata?.severity as string) || 'Unknown';
          severityDistribution[severity] = (severityDistribution[severity] || 0) + 1;
          totalConfidence += (e.metadata?.confidence as number) || 0;
        });

        // Most common result
        let mostCommonResult = 'N/A';
        let maxCount = 0;
        Object.entries(severityDistribution).forEach(([label, count]) => {
          if (count > maxCount) {
            mostCommonResult = label;
            maxCount = count;
          }
        });

        // Screenings per day
        const screeningsPerDay: Record<string, number> = {};
        completedEvents.forEach((e) => {
          const day = e.timestamp.slice(0, 10); // YYYY-MM-DD
          screeningsPerDay[day] = (screeningsPerDay[day] || 0) + 1;
        });

        // Last screening date
        const lastScreeningDate = completedEvents.length > 0
          ? completedEvents[completedEvents.length - 1].timestamp
          : null;

        return {
          totalScreenings: completedEvents.length,
          severityDistribution,
          avgConfidence:
            completedEvents.length > 0
              ? totalConfidence / completedEvents.length
              : 0,
          screeningsPerDay,
          mostCommonResult,
          lastScreeningDate,
        };
      },

      clearEvents: () => set({ events: [] }),
    }),
    {
      name: 'bari-analytics-storage',
      storage: secureStorage,
      partialize: (state) => ({
        events: state.events,
      }),
    },
  ),
);
