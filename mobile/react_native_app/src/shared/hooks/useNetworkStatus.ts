/**
 * useNetworkStatus.ts — Cross-platform network connectivity hook
 *
 * - Web: uses `navigator.onLine` + online/offline events
 * - Native: uses expo-network for efficient connectivity detection
 *   with AppState-based refresh instead of wasteful polling
 */
import { useState, useEffect, useCallback } from 'react';
import { Platform, AppState } from 'react-native';

interface NetworkStatus {
  isConnected: boolean;
  lastChecked: Date;
}

/**
 * Cross-platform network status hook.
 * On native, checks connectivity when the app becomes active.
 * On web, listens for browser online/offline events.
 */
export function useNetworkStatus(): NetworkStatus {
  const [status, setStatus] = useState<NetworkStatus>({
    isConnected: true,
    lastChecked: new Date(),
  });

  const checkConnection = useCallback(async () => {
    if (Platform.OS === 'web') {
      setStatus({
        isConnected: navigator.onLine,
        lastChecked: new Date(),
      });
    } else {
      // On native, try a lightweight fetch to check connectivity
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        await fetch('https://clients3.google.com/generate_204', {
          method: 'HEAD',
          signal: controller.signal,
        });
        clearTimeout(timeout);
        setStatus({ isConnected: true, lastChecked: new Date() });
      } catch {
        setStatus({ isConnected: false, lastChecked: new Date() });
      }
    }
  }, []);

  useEffect(() => {
    checkConnection();

    if (Platform.OS === 'web') {
      const handleOnline = () =>
        setStatus({ isConnected: true, lastChecked: new Date() });
      const handleOffline = () =>
        setStatus({ isConnected: false, lastChecked: new Date() });

      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);

      return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
      };
    } else {
      // Native: check when app returns to foreground instead of polling
      const subscription = AppState.addEventListener('change', (nextState) => {
        if (nextState === 'active') {
          checkConnection();
        }
      });
      return () => subscription.remove();
    }
  }, [checkConnection]);

  return status;
}
