/**
 * app/auth.tsx — Authentication screen (Sign In / Sign Up)
 *
 * Baby-color themed email & password form with toggle between modes.
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../src/shared/theme';
import { Button, InputField, Card, LoadingOverlay, ErrorMessage } from '../src/shared/components';
import { signIn, signUp } from '../src/services/supabaseAuth';

export default function AuthScreen() {
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

    // Validation
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }

    if (isSignUp && password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }

    setIsLoading(true);
    try {
      if (isSignUp) {
        await signUp(email.trim(), password);
        setSuccessMessage(
          'Account created! Check your email to confirm, then sign in.',
        );
        setMode('signin');
      } else {
        await signIn(email.trim(), password);
        // Auth state listener in _layout.tsx will handle navigation
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [email, password, confirmPassword, isSignUp]);

  const toggleMode = () => {
    setMode((prev) => (prev === 'signin' ? 'signup' : 'signin'));
    setError(null);
    setSuccessMessage(null);
  };

  return (
    <View style={styles.wrapper}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          style={styles.container}
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.iconCircle}>
              <Ionicons name="heart-outline" size={40} color={colors.primary} />
            </View>
            <Text style={styles.title}>Bari Anemia Screening</Text>
            <Text style={styles.subtitle}>
              {isSignUp ? 'Create your account' : 'Welcome back'}
            </Text>
          </View>

          {/* Success message */}
          {successMessage && (
            <Card style={styles.successCard}>
              <View style={styles.successRow}>
                <Ionicons
                  name="checkmark-circle"
                  size={20}
                  color={colors.severityNormal}
                />
                <Text style={styles.successText}>{successMessage}</Text>
              </View>
            </Card>
          )}

          {/* Form */}
          <Card style={styles.formCard}>
            <InputField
              label="Email"
              value={email}
              onChangeText={setEmail}
              placeholder="you@example.com"
              keyboardType="email-address"
            />

            <InputField
              label="Password"
              value={password}
              onChangeText={setPassword}
              placeholder="At least 6 characters"
              secureTextEntry
            />

            {isSignUp && (
              <InputField
                label="Confirm Password"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholder="Re-enter password"
                secureTextEntry
              />
            )}

            {error && <ErrorMessage message={error} />}

            <View style={styles.submitContainer}>
              <Button
                title={isSignUp ? 'Create Account' : 'Sign In'}
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

          {/* Toggle link */}
          <View style={styles.toggleContainer}>
            <Text style={styles.toggleText}>
              {isSignUp
                ? 'Already have an account?'
                : "Don't have an account?"}
            </Text>
            <Button
              title={isSignUp ? 'Sign In' : 'Sign Up'}
              onPress={toggleMode}
              variant="outline"
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      <LoadingOverlay
        visible={isLoading}
        message={isSignUp ? 'Creating account…' : 'Signing in…'}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
    backgroundColor: colors.background,
  },
  flex: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingTop: spacing.xxl,
    paddingBottom: spacing.xxl,
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  iconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  title: {
    ...typography.title,
    color: colors.primaryDark,
    textAlign: 'center',
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
  formCard: {
    paddingVertical: spacing.lg,
  },
  successCard: {
    backgroundColor: colors.severityNormalBg,
    marginBottom: spacing.md,
  },
  successRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  successText: {
    ...typography.caption,
    color: colors.text,
    marginLeft: spacing.sm,
    flex: 1,
  },
  submitContainer: {
    marginTop: spacing.lg,
  },
  toggleContainer: {
    alignItems: 'center',
    marginTop: spacing.lg,
  },
  toggleText: {
    ...typography.caption,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
  },
});
