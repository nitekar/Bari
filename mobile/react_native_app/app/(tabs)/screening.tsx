/**
 * app/(tabs)/screening.tsx — Main screening form
 *
 * Uses shared useScreeningForm hook for form logic.
 * Supports Quick Screen (image-only) and Full Screen (image + clinical data).
 */
import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Image,
  TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../../src/shared/theme';
import {
  Button,
  InputField,
  Card,
  GenderToggle,
  LoadingOverlay,
  ErrorMessage,
} from '../../src/shared/components';
import { useTranslation } from '../../src/i18n';
import { useScreeningForm } from '../../src/shared/hooks/useScreeningForm';

export default function ScreeningScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const { state, actions, derived } = useScreeningForm();

  return (
    <View style={styles.wrapper}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {/* Mode Toggle */}
        <View style={styles.modeToggle}>
          <TouchableOpacity
            style={[styles.modeBtn, state.mode === 'quick' && styles.modeBtnActive]}
            onPress={() => actions.setMode('quick')}
            activeOpacity={0.8}
          >
            <Ionicons
              name="eye-outline"
              size={16}
              color={state.mode === 'quick' ? colors.white : colors.primary}
            />
            <Text style={[styles.modeBtnText, state.mode === 'quick' && styles.modeBtnTextActive]}>
              Quick Screen
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.modeBtn, state.mode === 'full' && styles.modeBtnActive]}
            onPress={() => actions.setMode('full')}
            activeOpacity={0.8}
          >
            <Ionicons
              name="clipboard-outline"
              size={16}
              color={state.mode === 'full' ? colors.white : colors.primary}
            />
            <Text style={[styles.modeBtnText, state.mode === 'full' && styles.modeBtnTextActive]}>
              Full Screen
            </Text>
          </TouchableOpacity>
        </View>

        {/* Mode description */}
        <Text style={styles.modeHint}>
          {state.mode === 'quick'
            ? 'Image only — detects Anemic / Non-Anemic'
            : 'Image + clinical data — classifies severity (Mild / Moderate / Severe)'}
        </Text>

        {/* Patient Info — always shown */}
        <Text style={styles.sectionTitle}>{t.screening.patientInfo}</Text>

        <InputField
          label={t.screening.patientName}
          value={state.patientName}
          onChangeText={actions.setPatientName}
          placeholder="e.g. Amara Uwase"
        />

        <InputField
          label={t.screening.location}
          value={state.patientLocation}
          onChangeText={actions.setPatientLocation}
          placeholder="e.g. Kigali, Rwanda"
        />

        {/* Clinical fields — Full Screen only */}
        {state.mode === 'full' && (
          <>
            <InputField
              label={t.screening.patientAge}
              value={state.age}
              onChangeText={actions.setAge}
              placeholder={t.screening.agePlaceholder}
              keyboardType="numeric"
            />

            <GenderToggle value={state.gender} onChange={actions.setGender} />

            <InputField
              label={`${t.screening.hemoglobin} — ${t.screening.optional}`}
              value={state.hbLevel}
              onChangeText={actions.setHbLevel}
              placeholder={t.screening.hbPlaceholder}
              keyboardType="decimal-pad"
            />
          </>
        )}

        {/* Image Section */}
        <Text style={styles.sectionTitle}>Conjunctiva Image</Text>

        {derived.imageUri ? (
          <Card>
            <Image source={{ uri: derived.imageUri }} style={styles.imagePreview} />
            <View style={styles.imageActions}>
              <Button
                title="Change Image"
                onPress={() => router.push('/image-capture')}
                variant="outline"
                icon={<Ionicons name="refresh" size={18} color={colors.primary} />}
              />
            </View>
          </Card>
        ) : (
          <Card style={styles.uploadCard}>
            <Ionicons name="cloud-upload-outline" size={40} color={colors.textLight} />
            <Text style={styles.uploadText}>No image selected</Text>
            <Button
              title="Upload Image"
              onPress={() => router.push('/image-capture')}
              variant="secondary"
              icon={<Ionicons name="image-outline" size={18} color={colors.text} />}
            />
          </Card>
        )}

        {/* Error */}
        {derived.error && <ErrorMessage message={derived.error} onRetry={actions.handleSubmit} />}

        {/* Submit */}
        <View style={styles.submitContainer}>
          <Button
            title={state.mode === 'quick' ? 'Quick Screen' : 'Run Full Screening'}
            onPress={actions.handleSubmit}
            variant="primary"
            loading={derived.isLoading}
            disabled={!derived.canSubmit}
            icon={
              !derived.isLoading ? (
                <Ionicons name="pulse-outline" size={20} color={colors.white} />
              ) : undefined
            }
          />
        </View>
      </ScrollView>

      <LoadingOverlay
        visible={derived.isLoading}
        message={state.mode === 'quick' ? 'Analyzing image…' : 'Analyzing patient data…'}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
  },
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: 120,
  },
  // ── Mode toggle ──
  modeToggle: {
    flexDirection: 'row',
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.md,
    padding: 4,
    marginBottom: spacing.sm,
  },
  modeBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
    gap: 6,
  },
  modeBtnActive: {
    backgroundColor: colors.primary,
  },
  modeBtnText: {
    ...typography.captionBold,
    color: colors.primary,
  },
  modeBtnTextActive: {
    color: colors.white,
  },
  modeHint: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    ...typography.subtitle,
    color: colors.text,
    marginBottom: spacing.md,
    marginTop: spacing.sm,
  },
  imagePreview: {
    width: '100%',
    height: 200,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  imageActions: {
    alignItems: 'center',
  },
  uploadCard: {
    alignItems: 'center',
    borderStyle: 'dashed',
    borderWidth: 2,
    borderColor: colors.border,
    backgroundColor: colors.surfaceElevated,
  },
  uploadText: {
    ...typography.body,
    color: colors.textLight,
    marginVertical: spacing.md,
  },
  submitContainer: {
    marginTop: spacing.lg,
  },
});
