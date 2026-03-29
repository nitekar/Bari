/**
 * useStore.test.ts — Unit tests for Zustand store actions
 */

// Mock SecureStore
jest.mock('expo-secure-store', () => ({
  getItemAsync: jest.fn().mockResolvedValue(null),
  setItemAsync: jest.fn().mockResolvedValue(undefined),
  deleteItemAsync: jest.fn().mockResolvedValue(undefined),
}));

// Mock supabaseDb
jest.mock('../../services/supabaseDb', () => ({
  saveScreeningResult: jest.fn().mockResolvedValue({}),
  getScreeningHistory: jest.fn().mockResolvedValue([]),
  saveSleepLog: jest.fn().mockResolvedValue(undefined),
  saveFeedingLog: jest.fn().mockResolvedValue(undefined),
}));

import { useStore } from '../useStore';

describe('useStore', () => {
  beforeEach(() => {
    // Reset store to initial state
    useStore.getState().reset();
    useStore.setState({
      history: [],
      sleepLogs: [],
      feedingLogs: [],
      offlineQueue: [],
      userId: null,
      role: null,
      language: 'en',
    });
  });

  describe('setUserId', () => {
    it('sets the user ID', () => {
      useStore.getState().setUserId('user-123');
      expect(useStore.getState().userId).toBe('user-123');
    });

    it('clears the user ID with null', () => {
      useStore.getState().setUserId('user-123');
      useStore.getState().setUserId(null);
      expect(useStore.getState().userId).toBeNull();
    });
  });

  describe('setRole', () => {
    it('sets the role', () => {
      useStore.getState().setRole('admin');
      expect(useStore.getState().role).toBe('admin');
    });
  });

  describe('setLanguage', () => {
    it('sets the language', () => {
      useStore.getState().setLanguage('fr');
      expect(useStore.getState().language).toBe('fr');
    });
  });

  describe('setResult', () => {
    it('sets result and clears error', () => {
      useStore.setState({ error: 'previous error' });
      const mockResult = {
        prediction: 'Mild',
        confidence: 0.85,
        class_probabilities: {},
        nutrition: 'Eat spinach',
        recommended_foods: ['Spinach'],
        referral_action: 'Follow up in 90 days',
      };
      useStore.getState().setResult(mockResult);

      expect(useStore.getState().result).toEqual(mockResult);
      expect(useStore.getState().error).toBeNull();
    });
  });

  describe('addToHistory', () => {
    it('adds a record to history', () => {
      const record = {
        id: 'rec-1',
        date: '2026-03-29T10:00:00Z',
        prediction: 'Mild',
        confidence: 0.85,
        mode: 'tabular' as const,
        age: 24,
        gender: 1,
      };
      useStore.getState().addToHistory(record);

      expect(useStore.getState().history).toHaveLength(1);
      expect(useStore.getState().history[0].id).toBe('rec-1');
    });

    it('prepends new records (newest first)', () => {
      const record1 = {
        id: 'rec-1',
        date: '2026-03-29T10:00:00Z',
        prediction: 'Mild',
        confidence: 0.8,
        mode: 'tabular' as const,
      };
      const record2 = {
        id: 'rec-2',
        date: '2026-03-29T11:00:00Z',
        prediction: 'Severe',
        confidence: 0.95,
        mode: 'image' as const,
      };

      useStore.getState().addToHistory(record1);
      useStore.getState().addToHistory(record2);

      expect(useStore.getState().history[0].id).toBe('rec-2');
      expect(useStore.getState().history[1].id).toBe('rec-1');
    });

    it('limits history to 100 records', () => {
      for (let i = 0; i < 110; i++) {
        useStore.getState().addToHistory({
          id: `rec-${i}`,
          date: new Date().toISOString(),
          prediction: 'Mild',
          confidence: 0.8,
          mode: 'tabular',
        });
      }
      expect(useStore.getState().history.length).toBeLessThanOrEqual(100);
    });
  });

  describe('clearHistory', () => {
    it('empties the history array', () => {
      useStore.getState().addToHistory({
        id: 'rec-1',
        date: new Date().toISOString(),
        prediction: 'Mild',
        confidence: 0.8,
        mode: 'tabular',
      });
      useStore.getState().clearHistory();

      expect(useStore.getState().history).toHaveLength(0);
    });
  });

  describe('addSleepLog', () => {
    it('adds a sleep log entry', () => {
      const log = {
        id: 'sleep-1',
        startTime: '21:00',
        endTime: '07:00',
        durationHours: 10,
        notes: 'Slept well',
        date: '2026-03-29',
      };
      useStore.getState().addSleepLog(log);

      expect(useStore.getState().sleepLogs).toHaveLength(1);
      expect(useStore.getState().sleepLogs[0].id).toBe('sleep-1');
    });
  });

  describe('addFeedingLog', () => {
    it('adds a feeding log entry', () => {
      const log = {
        id: 'feed-1',
        type: 'breastfeeding' as const,
        time: '08:00',
        notes: 'Morning feed',
        date: '2026-03-29',
      };
      useStore.getState().addFeedingLog(log);

      expect(useStore.getState().feedingLogs).toHaveLength(1);
      expect(useStore.getState().feedingLogs[0].type).toBe('breastfeeding');
    });
  });

  describe('offlineQueue', () => {
    it('adds and removes from offline queue', () => {
      const req = {
        id: 'q-1',
        endpoint: '/predict/tabular',
        method: 'POST' as const,
        body: { age: 12 },
        contentType: 'application/json' as const,
        retryCount: 0,
        maxRetries: 3,
        nextRetryAt: new Date().toISOString(),
        createdAt: new Date().toISOString(),
      };

      useStore.getState().addToQueue(req);
      expect(useStore.getState().offlineQueue).toHaveLength(1);

      useStore.getState().removeFromQueue('q-1');
      expect(useStore.getState().offlineQueue).toHaveLength(0);
    });
  });

  describe('reset', () => {
    it('clears transient state without clearing history', () => {
      useStore.setState({
        result: { prediction: 'Mild', confidence: 0.8 } as any,
        imageUri: 'file:///img.jpg',
        isLoading: true,
        error: 'some error',
      });

      useStore.getState().reset();

      expect(useStore.getState().result).toBeNull();
      expect(useStore.getState().imageUri).toBeNull();
      expect(useStore.getState().isLoading).toBe(false);
      expect(useStore.getState().error).toBeNull();
    });
  });
});
