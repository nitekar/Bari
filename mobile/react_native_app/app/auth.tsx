/**
 * app/auth.tsx — Sign In / Sign Up / Forgot Password / Reset Password screen
 */
import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
} from 'react-native';
import * as SecureStore from 'expo-secure-store';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../src/shared/theme';
import { Button, InputField, Card, LoadingOverlay, ErrorMessage, Logo } from '../src/shared/components';
import {
  signIn,
  signUpWithRole,
  sendPasswordReset,
  updatePassword,
} from '../src/services/supabaseAuth';
import type { UserRole } from '../src/services/supabaseAuth';
import { isSupabaseConfigured } from '../src/services/supabase';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

type AuthMode = 'signin' | 'signup' | 'forgot' | 'newpassword';

// ── Role Picker ───────────────────────────────────────────────────────────────
interface RoleOption {
  role: UserRole;
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  description: string;
  color: string;
}

const ROLE_OPTIONS: RoleOption[] = [
  { role: 'chw',   icon: 'medkit-outline',  label: 'CHW',    description: 'Community Health Worker', color: colors.primary },
  { role: 'parent',icon: 'heart-outline',   label: 'Parent', description: 'Parent or Caregiver',  color: colors.secondary },
  { role: 'admin', icon: 'shield-outline',  label: 'Admin',  description: 'System Administrator', color: colors.error },
];

function RolePicker({
  selected,
  onChange,
}: {
  selected: UserRole;
  onChange: (r: UserRole) => void;
}) {
  return (
    <View style={rpStyles.container}>
      <Text style={rpStyles.label}>Select Role</Text>
      <View style={rpStyles.row}>
        {ROLE_OPTIONS.map((opt) => {
          const isSelected = selected === opt.role;
          return (
            <TouchableOpacity
              key={opt.role}
              style={[
                rpStyles.card,
                isSelected
                  ? { borderColor: colors.primaryDark, borderWidth: 2 }
                  : { borderColor: colors.border, borderWidth: 1 },
              ]}
              onPress={() => onChange(opt.role)}
              activeOpacity={0.7}
            >
              <View style={[rpStyles.iconBox, { backgroundColor: opt.color + '20' }]}>
                <Ionicons name={opt.icon} size={22} color={opt.color} />
              </View>
              <Text style={[rpStyles.roleLabel, isSelected && { color: colors.primaryDark }]}>
                {opt.label}
              </Text>
              <Text style={rpStyles.roleDesc}>{opt.description}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const rpStyles = StyleSheet.create({
  container: { marginBottom: spacing.md },
  label: { fontSize: 13, fontWeight: '600', color: colors.textSecondary, marginBottom: spacing.sm },
  row: { flexDirection: 'row', gap: spacing.sm },
  card: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xs,
    borderRadius: borderRadius.md,
    backgroundColor: colors.surfaceElevated,
  },
  iconBox: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  roleLabel: { fontSize: 13, fontWeight: '700', color: colors.text, marginBottom: 2 },
  roleDesc: { fontSize: 10, color: colors.textSecondary, textAlign: 'center', lineHeight: 13 },
});

// ── Auth Screen ───────────────────────────────────────────────────────────────
export default function AuthScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ mode?: string }>();
  const { t } = useTranslation();
  const setUserId = useStore((s) => s.setUserId);
  const hasAcceptedEula = useStore((s) => s.hasAcceptedEula);

  const [mode, setMode] = useState<AuthMode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [selectedRole, setSelectedRole] = useState<UserRole>('chw');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [rememberMe, setRememberMe] = useState(false);

  // Load saved credentials
  useEffect(() => {
    SecureStore.getItemAsync('bari_credentials').then((res) => {
      if (res) {
        try {
          const creds = JSON.parse(res);
          if (creds.email && creds.password) {
            setEmail(creds.email);
            setPassword(creds.password);
            setRememberMe(true);
          }
        } catch (e) {}
      }
    });
  }, []);

  // Deep-link: anemia-screening://reset-password?mode=newpassword
  useEffect(() => {
    if (params.mode === 'newpassword') {
      setMode('newpassword');
      setError(null);
      setSuccessMessage(null);
    }
  }, [params.mode]);

  const switchMode = (next: AuthMode) => {
    setMode(next);
    setError(null);
    setSuccessMessage(null);
  };

  // ── Submit handlers ──────────────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    setError(null);
    setSuccessMessage(null);
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    // Forgot password
    if (mode === 'forgot') {
      if (!email.trim()) { setError('Please enter your email address.'); return; }
      if (!emailRegex.test(email.trim())) { setError(t.auth.invalidEmail); return; }
      setIsLoading(true);
      try {
        await sendPasswordReset(email.trim());
        setSuccessMessage('Password reset link sent — check your email.');
        switchMode('signin');
      } catch (err: any) {
        setError(err.message || 'Failed to send reset email.');
      } finally {
        setIsLoading(false);
      }
      return;
    }

    // New password (from deep link)
    if (mode === 'newpassword') {
      if (!password.trim()) { setError('Please enter a new password.'); return; }
      if (password.length < 6) { setError(t.auth.minPassword); return; }
      if (password !== confirmPassword) { setError(t.auth.passwordMismatch); return; }
      setIsLoading(true);
      try {
        await updatePassword(password);
        setSuccessMessage('Password updated successfully. Please sign in.');
        switchMode('signin');
      } catch (err: any) {
        setError(err.message || 'Failed to update password.');
      } finally {
        setIsLoading(false);
      }
      return;
    }

    // Sign in / Sign up
    if (!email.trim() || !password.trim()) { setError(t.auth.fillBoth); return; }
    if (!emailRegex.test(email.trim())) { setError(t.auth.invalidEmail); return; }
    if (mode === 'signup' && password !== confirmPassword) { setError(t.auth.passwordMismatch); return; }
    if (password.length < 6) { setError(t.auth.minPassword); return; }

    setIsLoading(true);
    try {
      if (mode === 'signup') {
        if (rememberMe) {
          SecureStore.setItemAsync('bari_credentials', JSON.stringify({ email: email.trim(), password })).catch(() => {});
        } else {
          SecureStore.deleteItemAsync('bari_credentials').catch(() => {});
        }
        await signUpWithRole(email.trim(), password, selectedRole);
        setSuccessMessage(t.auth.confirmEmail);
        switchMode('signin');
      } else {
        if (rememberMe) {
          SecureStore.setItemAsync('bari_credentials', JSON.stringify({ email: email.trim(), password })).catch(() => {});
        } else {
          SecureStore.deleteItemAsync('bari_credentials').catch(() => {});
        }
        await signIn(email.trim(), password);
        // Auth state listener in _layout.tsx navigates based on role
      }
    } catch (err: any) {
      setError(err.message || t.auth.failed);
    } finally {
      setIsLoading(false);
    }
  }, [email, password, confirmPassword, mode, selectedRole, t]);

  // Guest mode
  const handleContinueAsGuest = () => {
    setUserId(null); // Triggers layout interceptor correctly for guest identity
    if (!hasAcceptedEula) {
      router.replace('/eula-welcome');
    } else {
      router.replace('/');
    }
  };

  // ── Derived display ──────────────────────────────────────────────────────
  const headerSubtitle = {
    signin: t.auth.welcomeBack,
    signup: t.auth.createAccount,
    forgot: 'Reset your password',
    newpassword: 'Set a new password',
  }[mode];

  const submitLabel = {
    signin: t.auth.signIn,
    signup: t.auth.createAccountBtn,
    forgot: 'Send Reset Link',
    newpassword: 'Update Password',
  }[mode];

  const submitIcon = {
    signin: 'log-in-outline',
    signup: 'person-add-outline',
    forgot: 'mail-outline',
    newpassword: 'lock-closed-outline',
  }[mode] as keyof typeof Ionicons.glyphMap;

  const loadingLabel = {
    signin: t.auth.signingIn,
    signup: t.auth.creatingAccount,
    forgot: 'Sending…',
    newpassword: 'Updating…',
  }[mode];

  const showGuestOption = mode === 'signin' || mode === 'signup';

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
            <Text style={styles.subtitle}>{headerSubtitle}</Text>
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
            {/* Email field — not shown on newpassword */}
            {mode !== 'newpassword' && (
              <InputField
                label={t.auth.email}
                value={email}
                onChangeText={setEmail}
                placeholder="you@example.com"
                keyboardType="email-address"
              />
            )}

            {/* Password field — not shown on forgot */}
            {mode !== 'forgot' && (
              <InputField
                label={mode === 'newpassword' ? 'New Password' : t.auth.password}
                value={password}
                onChangeText={setPassword}
                placeholder="••••••••"
                secureTextEntry
              />
            )}

            {/* Confirm password — signup and newpassword */}
            {(mode === 'signup' || mode === 'newpassword') && (
              <InputField
                label={t.auth.confirmPassword}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholder="••••••••"
                secureTextEntry
              />
            )}

            {/* Role picker — only signup */}
            {mode === 'signup' && (
              <RolePicker selected={selectedRole} onChange={setSelectedRole} />
            )}

            {/* Forgot password link — only signin */}
            {mode === 'signin' && (
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: -spacing.xs, marginBottom: spacing.sm }}>
                <TouchableOpacity
                  style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 4 }}
                  onPress={() => setRememberMe(!rememberMe)}
                  activeOpacity={0.7}
                >
                  <Ionicons
                    name={rememberMe ? 'checkbox' : 'square-outline'}
                    size={20}
                    color={rememberMe ? colors.primaryDark : colors.border}
                  />
                  <Text style={{ marginLeft: 8, fontSize: 13, color: colors.textSecondary }}>Remember me</Text>
                </TouchableOpacity>

                <TouchableOpacity onPress={() => switchMode('forgot')}>
                  <Text style={styles.forgotText}>Forgot password?</Text>
                </TouchableOpacity>
              </View>
            )}

            {error && <ErrorMessage message={error} />}

            <View style={styles.submitWrap}>
              <Button
                title={submitLabel}
                onPress={handleSubmit}
                variant="primary"
                loading={isLoading}
                icon={
                  !isLoading ? (
                    <Ionicons name={submitIcon} size={20} color={colors.white} />
                  ) : undefined
                }
              />
            </View>

            {/* Back to Sign In — forgot and newpassword */}
            {(mode === 'forgot' || mode === 'newpassword') && (
              <TouchableOpacity
                onPress={() => switchMode('signin')}
                style={styles.backBtn}
              >
                <Ionicons name="arrow-back-outline" size={16} color={colors.primaryDark} />
                <Text style={styles.backText}>Back to Sign In</Text>
              </TouchableOpacity>
            )}
          </Card>

          {/* ── Toggle sign-in / sign-up ── */}
          {showGuestOption && (
            <View style={styles.toggleRow}>
              <Text style={styles.toggleText}>
                {mode === 'signup' ? t.auth.hasAccount : t.auth.noAccount}
              </Text>
              <TouchableOpacity
                onPress={() => switchMode(mode === 'signup' ? 'signin' : 'signup')}
                style={styles.toggleBtn}
              >
                <Text style={styles.toggleLink}>
                  {mode === 'signup' ? t.auth.signIn : t.auth.signUp}
                </Text>
              </TouchableOpacity>
            </View>
          )}

          {/* ── Divider + Guest ── */}
          {showGuestOption && (
            <>
              <View style={styles.dividerRow}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>or</Text>
                <View style={styles.dividerLine} />
              </View>

              <Button
                title="Continue as Guest"
                onPress={handleContinueAsGuest}
                variant="secondary"
                icon={<Ionicons name="person-outline" size={20} color={colors.text} />}
              />
            </>
          )}

          {!isSupabaseConfigured && (
            <Text style={styles.devNote}>
              Supabase not configured — sign-in will fail. Use guest mode for testing.
            </Text>
          )}
        </ScrollView>
      </KeyboardAvoidingView>

      <LoadingOverlay visible={isLoading} message={loadingLabel} />
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
  forgotBtn: { alignSelf: 'flex-end', marginTop: -spacing.xs, marginBottom: spacing.sm },
  forgotText: { fontSize: 13, color: colors.primaryDark, fontWeight: '600' },
  submitWrap: { marginTop: spacing.lg },
  backBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.md,
    gap: spacing.xs,
  },
  backText: { fontSize: 14, color: colors.primaryDark, fontWeight: '600' },
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
