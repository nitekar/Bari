/**
 * app/auth.tsx — Sign In / Sign Up screen
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../src/shared/theme';
import { Button, InputField, Card, LoadingOverlay, ErrorMessage, Logo } from '../src/shared/components';
import { signIn, signUp } from '../src/services/supabaseAuth';
import { isSupabaseConfigured } from '../src/services/supabase';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

export default function AuthScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const setUserId = useStore((s) => s.setUserId);

  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const isSignUp = mode === 'signup';

  const handleSubmit = useCallback(async () => {
    setError(null);
    setSuccessMessage(null);

    if (!email.trim() || !password.trim()) {
      setError(t.auth.fillBoth);
      return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      setError(t.auth.invalidEmail);
      return;
    }
    if (isSignUp && password !== confirmPassword) {
      setError(t.auth.passwordMismatch);
      return;
    }
    if (password.length < 6) {
      setError(t.auth.minPassword);
      return;
    }

    setIsLoading(true);
    try {
      if (isSignUp) {
        await signUp(email.trim(), password);
        setSuccessMessage(t.auth.confirmEmail);
        setMode('signin');
      } else {
        await signIn(email.trim(), password);
        // Auth state listener in _layout.tsx navigates to '/'
      }
    } catch (err: any) {
      setError(err.message || t.auth.failed);
    } finally {
      setIsLoading(false);
    }
  }, [email, password, confirmPassword, isSignUp, t]);

  const toggleMode = () => {
    setMode((prev) => (prev === 'signin' ? 'signup' : 'signin'));
    setError(null);
    setSuccessMessage(null);
  };

  // Guest mode — skip auth and go straight to the app
  const handleContinueAsGuest = () => {
    setUserId(null);   // ensure no stale userId
    router.replace('/');
  };

  return (
    <View style={styles.wrapper}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* ── Header ── */}
          <View style={styles.header}>
            <View style={styles.bubble1} />
            <View style={styles.bubble2} />
            <Logo size="lg" horizontal={false} showText />
            <Text style={styles.subtitle}>
              {isSignUp ? t.auth.createAccount : t.auth.welcomeBack}
            </Text>
          </View>

          {/* ── Success banner ── */}
          {successMessage && (
            <Card style={styles.successCard}>
              <View style={styles.successRow}>
                <Ionicons name="checkmark-circle" size={20} color={colors.severityNormal} />
                <Text style={styles.successText}>{successMessage}</Text>
              </View>
            </Card>
          )}

          {/* ── Form ── */}
          <Card style={styles.formCard}>
            <InputField
              label={t.auth.email}
              value={email}
              onChangeText={setEmail}
              placeholder="you@example.com"
              keyboardType="email-address"
              autoCapitalize="none"
            />
            <InputField
              label={t.auth.password}
              value={password}
              onChangeText={setPassword}
              placeholder="••••••••"
              secureTextEntry
            />
            {isSignUp && (
              <InputField
                label={t.auth.confirmPassword}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholder="••••••••"
                secureTextEntry
              />
            )}

            {error && <ErrorMessage message={error} />}

            <View style={styles.submitWrap}>
              <Button
                title={isSignUp ? t.auth.createAccountBtn : t.auth.signIn}
                onPress={handleSubmit}
                variant="primary"
                loading={isLoading}
                icon={
                  !isLoading ? (
                    <Ionicons
                      name={isSignUp ? 'person-add-outline' : 'log-in-outline'}
                      size={20}
                      color={colors.white}
                    />
                  ) : undefined
                }
              />
            </View>
          </Card>

          {/* ── Toggle sign-in / sign-up ── */}
          <View style={styles.toggleRow}>
            <Text style={styles.toggleText}>
              {isSignUp ? t.auth.hasAccount : t.auth.noAccount}
            </Text>
            <TouchableOpacity onPress={toggleMode} style={styles.toggleBtn}>
              <Text style={styles.toggleLink}>
                {isSignUp ? t.auth.signIn : t.auth.signUp}
              </Text>
            </TouchableOpacity>
          </View>

          {/* ── Divider ── */}
          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* ── Continue as Guest ── */}
          <Button
            title="Continue as Guest"
            onPress={handleContinueAsGuest}
            variant="secondary"
            icon={<Ionicons name="person-outline" size={20} color={colors.text} />}
          />

          {!isSupabaseConfigured && (
            <Text style={styles.devNote}>
              ⚙️  Supabase not configured — sign-in will fail. Use guest mode for testing.
            </Text>
          )}
        </ScrollView>
      </KeyboardAvoidingView>

      <LoadingOverlay
        visible={isLoading}
        message={isSignUp ? t.auth.creatingAccount : t.auth.signingIn}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  scroll: { flex: 1 },
  content: {
    padding: spacing.lg,
    paddingTop: 0,
    paddingBottom: spacing.xxl,
  },
  // ── Header ──
  header: {
    alignItems: 'center',
    paddingTop: 64,
    paddingBottom: 32,
    position: 'relative',
    overflow: 'hidden',
  },
  bubble1: {
    position: 'absolute',
    top: -20,
    right: -30,
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: colors.secondary + '35',
  },
  bubble2: {
    position: 'absolute',
    top: 10,
    left: -40,
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: colors.accent + '40',
  },
  subtitle: {
    fontSize: 16,
    color: colors.textSecondary,
    marginTop: 12,
    fontWeight: '500',
  },
  // ── Success ──
  successCard: {
    backgroundColor: colors.severityNormalBg,
    marginBottom: spacing.md,
  },
  successRow: { flexDirection: 'row', alignItems: 'center' },
  successText: {
    fontSize: 13,
    color: colors.text,
    marginLeft: spacing.sm,
    flex: 1,
  },
  // ── Form ──
  formCard: { paddingVertical: spacing.lg, marginBottom: spacing.md },
  submitWrap: { marginTop: spacing.lg },
  // ── Toggle ──
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  toggleText: { fontSize: 14, color: colors.textSecondary },
  toggleBtn: { paddingVertical: 4 },
  toggleLink: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.primaryDark,
  },
  // ── Divider ──
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.lg,
    gap: spacing.md,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: colors.border },
  dividerText: { fontSize: 13, color: colors.textLight },
  // ── Dev note ──
  devNote: {
    marginTop: spacing.lg,
    fontSize: 12,
    color: colors.textLight,
    textAlign: 'center',
    fontStyle: 'italic',
    lineHeight: 18,
  },
});
