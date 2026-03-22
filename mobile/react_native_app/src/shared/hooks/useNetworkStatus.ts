/**
 * useNetworkStatus.ts — Cross-platform network connectivity hook
 */
import { useState, useEffect, useCallback } from 'react';
import { Platform } from 'react-native';

interface NetworkStatus {
  isConnected: boolean;
  lastChecked: Date;
}

/**
 * Simple network status hook that works on both native and web.
 *
 * - Web: uses `navigator.onLine` + online/offline events
 * - Native: uses a periodic connectivity check via fetch to the API
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
      // Native: check every 30 seconds
      const interval = setInterval(checkConnection, 30_000);
      return () => clearInterval(interval);
    }
  }, [checkConnection]);

  return status;
}
