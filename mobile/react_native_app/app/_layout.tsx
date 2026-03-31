/**
 * app/_layout.tsx — Root Stack layout
 *
 * Handles:
 *  - Onboarding redirect (first-time users)
 *  - Supabase auth guard (when configured)
 *  - Role-based routing: admin → /(admin), chw → /(chw), parent → /(parent)
 *  - EMAIL_CONFIRMED event alert
 *  - Deep link token exchange for email confirmation
 *  - ErrorBoundary for unhandled render errors
 *  - Branded loading splash
 */
import React, { useEffect, useRef, useState } from 'react';
import { Alert, View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as Linking from 'expo-linking';
import type { Session } from '@supabase/supabase-js';
import { colors } from '../src/shared/theme';
import { onAuthStateChanged, getSession, getUserProfile } from '../src/services/supabaseAuth';
import type { UserRole } from '../src/services/supabaseAuth';
import { useNetworkStatus } from '../src/shared/hooks/useNetworkStatus';
import { ErrorBoundary as AppErrorBoundary } from '../src/shared/components';
import OfflineIndicator from '../src/shared/components/OfflineIndicator';
import { processSyncQueue } from '../src/services/syncService';
import { isSupabaseConfigured, supabase } from '../src/services/supabase';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

// ── Expo Router ErrorBoundary export ─────────────────────────────────────────
export { default as ErrorBoundary } from '../src/shared/components/ErrorBoundary';

// ── Branded loading splash ───────────────────────────────────────────────────
function LoadingSplash() {
  return (
    <View style={splashStyles.container}>
      <Text style={splashStyles.logo}>🩺</Text>
      <Text style={splashStyles.title}>Bari Anemia</Text>
      <Text style={splashStyles.subtitle}>Loading…</Text>
      <ActivityIndicator
        size="small"
        color={colors.primary}
        style={splashStyles.spinner}
      />
    </View>
  );
}

const splashStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: { fontSize: 48, marginBottom: 12 },
  title: {
    fontSize: 24,
    fontWeight: '800',
    color: colors.primaryDark,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: 4,
  },
  spinner: { marginTop: 24 },
});

export default function RootLayout() {
  const { t } = useTranslation();
  const [session, setSession] = useState<Session | null>(null);
  const [role, setRoleLocal] = useState<UserRole | null>(null);
  const [isReady, setIsReady] = useState(false);
  const router = useRouter();
  const segments = useSegments();
  const setUserId = useStore((s) => s.setUserId);
  const setRole = useStore((s) => s.setRole);
  const loadHistoryFromSupabase = useStore((s) => s.loadHistoryFromSupabase);
  const hasSeenOnboarding = useStore((s) => s.hasSeenOnboarding);
  const offlineQueue = useStore((s) => s.offlineQueue);
  const lastSyncedAt = useStore((s) => s.lastSyncedAt);

  const { isConnected } = useNetworkStatus();
  const prevConnected = useRef(isConnected);

  // ── Bootstrap: check existing session ──────────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) { setIsReady(true); return; }
    getSession().then(async (s) => {
      setSession(s);
      if (s?.user) {
        setUserId(s.user.id);
        loadHistoryFromSupabase(s.user.id);
        const profile = await getUserProfile();
        const r = profile?.role ?? 'chw';
        setRoleLocal(r);
        setRole(r);
      }
      setIsReady(true);
    });
  }, []);

  // ── Listen for auth state changes ──────────────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) return;
    const unsubscribe = onAuthStateChanged(async (event, s) => {
      if ((event as string) === 'EMAIL_CONFIRMED') {
        Alert.alert('Email Confirmed', 'Your email has been confirmed. You can now sign in.');
      }
      setSession(s);
      if (s?.user) {
        setUserId(s.user.id);
        loadHistoryFromSupabase(s.user.id);
        const profile = await getUserProfile();
        const r = profile?.role ?? 'chw';
        setRoleLocal(r);
        setRole(r);
      } else {
        setUserId(null);
        setRoleLocal(null);
        setRole(null);
      }
    });
    return unsubscribe;
  }, []);

  // ── Deep link handler: email confirmation ──────────────────────────────────
  useEffect(() => {
    if (!isSupabaseConfigured) return;
    const isValidJwtFormat = (token: string) =>
      /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token);

    const handleUrl = async (url: string) => {
      const fragment = url.split('#')[1] ?? '';
      const params = new URLSearchParams(fragment);
      const access_token = params.get('access_token');
      const refresh_token = params.get('refresh_token');
      // Validate token format before accepting
      if (
        access_token && refresh_token &&
        isValidJwtFormat(access_token) && isValidJwtFormat(refresh_token)
      ) {
        await supabase.auth.setSession({ access_token, refresh_token });
      }
    };
    Linking.getInitialURL().then((url) => { if (url) handleUrl(url); });
    const sub = Linking.addEventListener('url', ({ url }) => handleUrl(url));
    return () => sub.remove();
  }, []);

  // ── Sync offline queue when connectivity is restored ──────────────────────
  useEffect(() => {
    if (isConnected && !prevConnected.current) {
      processSyncQueue();
    }
    prevConnected.current = isConnected;
  }, [isConnected]);

  // ── Role-based route protection ────────────────────────────────────────────
  useEffect(() => {
    if (!isReady) return;
    const inTabs     = segments[0] === '(tabs)';
    const inChw      = segments[0] === '(chw)';
    const inAdmin    = segments[0] === '(admin)';
    const inParent   = segments[0] === '(parent)';
    const onAuth     = segments[0] === 'auth';
    const onBoarding = segments[0] === 'onboarding';

    if (!hasSeenOnboarding && !onBoarding) { router.replace('/onboarding'); return; }

    if (hasSeenOnboarding) {
      if (!session && !onAuth && !onBoarding) { router.replace('/auth'); return; }
      if (session && role) {
        if (onAuth) {
          if (role === 'admin')  { router.replace('/(admin)'); return; }
          if (role === 'parent') { router.replace('/(parent)'); return; }
          router.replace('/(chw)'); return;
        }
        // Redirect CHW users away from legacy (tabs) into (chw)
        if (inTabs && role === 'chw') { router.replace('/(chw)'); return; }
        // Redirect users to their role-group if mismatched
        if (inTabs && role === 'admin')  { router.replace('/(admin)'); return; }
        if (inTabs && role === 'parent') { router.replace('/(parent)'); return; }
      }
    }
  }, [session, role, segments, isReady, hasSeenOnboarding]);

  if (!isReady) return <LoadingSplash />;

  return (
    <AppErrorBoundary>
    <SafeAreaProvider>
      <StatusBar style="dark" backgroundColor={colors.background} />
      <View style={{ flex: 1 }}>
        <OfflineIndicator
          isConnected={isConnected}
          queueCount={offlineQueue.length}
          lastSyncedAt={lastSyncedAt ? new Date(lastSyncedAt) : null}
        />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.surface },
          headerTintColor: colors.primaryDark,
          headerTitleStyle: { fontWeight: '700', fontSize: 17, color: colors.text },
          headerShadowVisible: false,
          contentStyle: { backgroundColor: colors.background },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="onboarding"    options={{ headerShown: false }} />
        <Stack.Screen name="auth"          options={{ headerShown: false }} />
        <Stack.Screen name="(tabs)"        options={{ headerShown: false }} />
        <Stack.Screen name="(chw)"         options={{ headerShown: false }} />
        <Stack.Screen name="(admin)"       options={{ headerShown: false }} />
        <Stack.Screen name="(parent)"      options={{ headerShown: false }} />
        <Stack.Screen name="image-capture" options={{ title: t.tabs.screening }} />
        <Stack.Screen name="result"        options={{ title: 'Results' }} />
        <Stack.Screen name="referral"      options={{ title: t.referral.referralLetter }} />
        <Stack.Screen name="settings"      options={{ title: t.settings.title }} />
        <Stack.Screen name="legal"         options={{ title: 'Legal' }} />
        <Stack.Screen name="parent-sleep"       options={{ title: 'Sleep Log' }} />
        <Stack.Screen name="parent-feeding"     options={{ title: 'Feeding Log' }} />
        <Stack.Screen name="parent-development" options={{ title: 'Development Stage' }} />
      </Stack>
      </View>
    </SafeAreaProvider>
    </AppErrorBoundary>
  );
}
