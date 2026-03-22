/**
 * app/_layout.tsx — Root layout with Expo Router Stack
 *
 * Listens for Supabase auth state changes and redirects
 * unauthenticated users to the /auth screen.
 */
import React, { useEffect, useState } from 'react';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import type { Session } from '@supabase/supabase-js';
import { colors, typography } from '../src/shared/theme';
import { onAuthStateChanged, getSession } from '../src/services/supabaseAuth';
import { useStore } from '../src/app/store/useStore';

export default function RootLayout() {
  const [session, setSession] = useState<Session | null>(null);
  const [isReady, setIsReady] = useState(false);
  const router = useRouter();
  const segments = useSegments();
  const setUserId = useStore((s) => s.setUserId);
  const loadHistoryFromSupabase = useStore((s) => s.loadHistoryFromSupabase);

  // ── Bootstrap: check existing session ──────────────────────────────────────
  useEffect(() => {
    getSession().then((s) => {
      setSession(s);
      if (s?.user) {
        setUserId(s.user.id);
        loadHistoryFromSupabase(s.user.id);
      }
      setIsReady(true);
    });
  }, []);

  // ── Listen for auth changes ────────────────────────────────────────────────
  useEffect(() => {
    const unsubscribe = onAuthStateChanged((_event, s) => {
      setSession(s);
      if (s?.user) {
        setUserId(s.user.id);
        loadHistoryFromSupabase(s.user.id);
      } else {
        setUserId(null);
      }
    });
    return unsubscribe;
  }, []);

  // ── Route protection ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isReady) return;

    const onAuthScreen = segments[0] === 'auth';

    if (!session && !onAuthScreen) {
      router.replace('/auth');
    } else if (session && onAuthScreen) {
      router.replace('/');
    }
  }, [session, segments, isReady]);

  // Don't render anything until we know the auth state
  if (!isReady) return null;

  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: colors.primary,
          },
          headerTintColor: colors.white,
          headerTitleStyle: {
            ...typography.subtitle,
            color: colors.white,
          },
          headerShadowVisible: false,
          contentStyle: {
            backgroundColor: colors.background,
          },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen
          name="auth"
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="index"
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="screening"
          options={{ title: 'Screening' }}
        />
        <Stack.Screen
          name="image-capture"
          options={{ title: 'Select Image' }}
        />
        <Stack.Screen
          name="result"
          options={{ title: 'Results' }}
        />
        <Stack.Screen
          name="education"
          options={{ title: 'Learn About Anemia' }}
        />
        <Stack.Screen
          name="analytics"
          options={{ title: 'Analytics' }}
        />
        <Stack.Screen
          name="referral"
          options={{ title: 'Doctor Referral' }}
        />
      </Stack>
    </SafeAreaProvider>
  );
}
