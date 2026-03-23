/**
 * app/_layout.tsx — Root Stack layout
 *
 * Handles:
 *  - Onboarding redirect (first-time users)
 *  - Supabase auth guard (when configured)
 *  - Registers all routes: (tabs) group + stack screens
 */
import React, { useEffect, useState } from 'react';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as Linking from 'expo-linking';
import type { Session } from '@supabase/supabase-js';
import { colors } from '../src/shared/theme';
import { onAuthStateChanged, getSession } from '../src/services/supabaseAuth';
import { isSupabaseConfigured, supabase } from '../src/services/supabase';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

export default function RootLayout() {
  const { t } = useTranslation();
  const [session, setSession] = useState<Session | null>(null);
  const [isReady, setIsReady] = useState(false);
  const router = useRouter();
  const segments = useSegments();
  const setUserId = useStore((s) => s.setUserId);
  const loadHistoryFromSupabase = useStore((s) => s.loadHistoryFromSupabase);
  const hasSeenOnboarding = useStore((s) => s.hasSeenOnboarding);

  // ── Bootstrap: check existing session ──────────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) {
      setIsReady(true);
      return;
    }
    getSession().then((s) => {
      setSession(s);
      if (s?.user) {
        setUserId(s.user.id);
        loadHistoryFromSupabase(s.user.id);
      }
      setIsReady(true);
    });
  }, []);

  // ── Listen for auth state changes ──────────────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) return;
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

  // ── Deep link handler: email confirmation ──────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) return;

    const handleUrl = async (url: string) => {
      // Supabase appends tokens in the URL fragment: anemia-screening://#access_token=...
      const fragment = url.split('#')[1] ?? '';
      const params = new URLSearchParams(fragment);
      const access_token = params.get('access_token');
      const refresh_token = params.get('refresh_token');
      if (access_token && refresh_token) {
        await supabase.auth.setSession({ access_token, refresh_token });
      }
    };

    // App opened from a cold start via deep link
    Linking.getInitialURL().then((url) => { if (url) handleUrl(url); });

    // App already open and receives a deep link
    const sub = Linking.addEventListener('url', ({ url }) => handleUrl(url));
    return () => sub.remove();
  }, []);

  // ── Route protection ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isReady) return;

    const onOnboarding = segments[0] === 'onboarding';
    const onAuthScreen = segments[0] === 'auth';

    // 1. Onboarding gate
    if (!hasSeenOnboarding && !onOnboarding) {
      router.replace('/onboarding');
      return;
    }

    // 2. Auth gate — always shown after onboarding; auth actions only work
    //    once real Supabase credentials are configured.
    if (hasSeenOnboarding) {
      if (!session && !onAuthScreen && !onOnboarding) {
        router.replace('/auth');
      } else if (session && onAuthScreen) {
        router.replace('/');
      }
    }
  }, [session, segments, isReady, hasSeenOnboarding]);

  if (!isReady) return null;

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" backgroundColor={colors.background} />
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: colors.surface,
          },
          headerTintColor: colors.primaryDark,
          headerTitleStyle: {
            fontWeight: '700',
            fontSize: 17,
            color: colors.text,
          },
          headerShadowVisible: false,
          contentStyle: { backgroundColor: colors.background },
          animation: 'slide_from_right',
        }}
      >
        {/* Full-screen flows — no header */}
        <Stack.Screen name="onboarding" options={{ headerShown: false }} />
        <Stack.Screen name="auth"       options={{ headerShown: false }} />

        {/* Tab group — tab bar is its own chrome */}
        <Stack.Screen name="(tabs)"     options={{ headerShown: false }} />

        {/* Stack screens pushed on top of tabs */}
        <Stack.Screen name="image-capture" options={{ title: t.tabs.screening }} />
        <Stack.Screen name="result"        options={{ title: 'Results' }} />
        <Stack.Screen name="referral"      options={{ title: t.referral.referralLetter }} />
        <Stack.Screen name="settings"      options={{ title: t.settings.title }} />
      </Stack>
    </SafeAreaProvider>
  );
}
