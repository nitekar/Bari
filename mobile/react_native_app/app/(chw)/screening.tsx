/**
 * app/(chw)/screening.tsx — Professional screening form for CHW role
 *
 * Uses shared useScreeningForm hook for form logic.
 * Two modes:
 *   Quick Screen — image only → Anemic / Non-Anemic (binary)
 *   Full Screen  — image + clinical data → 4-class severity
 *
 * Layout: step cards + fixed bottom action bar
 */
import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Image,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, typography, spacing, borderRadius } from '../../src/shared/theme';
import {
  InputField,
  GenderToggle,
  LoadingOverlay,
} from '../../src/shared/components';
import { useTranslation } from '../../src/i18n';
import { useScreeningForm } from '../../src/shared/hooks/useScreeningForm';

// ── Step header ───────────────────────────────────────────────────────────────
function StepHeader({
  step,
  title,
  subtitle,
  complete,
}: {
  step: number;
  title: string;
  subtitle?: string;
  complete?: boolean;
}) {
  return (
    <View style={styles.stepHeader}>
      <View style={[styles.stepBadge, complete && styles.stepBadgeComplete]}>
        {complete ? (
          <Ionicons name="checkmark" size={14} color={colors.white} />
        ) : (
          <Text style={styles.stepNumber}>{step}</Text>
        )}
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.stepTitle}>{title}</Text>
        {subtitle && <Text style={styles.stepSubtitle}>{subtitle}</Text>}
      </View>
    </View>
  );
}

// ── Section card ──────────────────────────────────────────────────────────────
function SectionCard({ children }: { children: React.ReactNode }) {
  return <View style={styles.sectionCard}>{children}</View>;
}

// ── Mode pill ─────────────────────────────────────────────────────────────────
function ModePill({
  label,
  icon,
  description,
  active,
  onPress,
}: {
  label: string;
  icon: string;
  description: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.modePill, active && styles.modePillActive]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <Ionicons name={icon as any} size={20} color={active ? colors.white : colors.primary} />
      <View style={{ flex: 1, marginLeft: spacing.sm }}>
        <Text style={[styles.modePillLabel, active && styles.modePillLabelActive]}>
          {label}
        </Text>
        <Text style={[styles.modePillDesc, active && styles.modePillDescActive]}>
          {description}
        </Text>
      </View>
      {active && <Ionicons name="checkmark-circle" size={18} color={colors.white} />}
    </TouchableOpacity>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function ScreeningScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const { state, actions, derived } = useScreeningForm();

  const steps = state.mode === 'quick' ? [derived.hasImage] : [true, derived.hasAge, derived.hasImage];
  const completedSteps = steps.filter(Boolean).length;
  const totalSteps = steps.length;

  return (
    <View style={styles.root}>
      {/* ── Scrollable body ─────────────────────────────────────────────── */}
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[styles.content, { paddingBottom: 120 + insets.bottom }]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* Progress */}
        <View style={styles.progressRow}>
          <Text style={styles.progressLabel}>
            {completedSteps} of {totalSteps} {completedSteps === totalSteps ? '· Ready' : '· In progress'}
          </Text>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                { width: `${(completedSteps / totalSteps) * 100}%` as any },
              ]}
            />
          </View>
        </View>

        {/* Step 1 — Mode */}
        <SectionCard>
          <StepHeader step={1} title="Screening Mode" subtitle="Choose based on available data" complete />
          <ModePill
            label="Quick Screen"
            icon="eye-outline"
            description="Image only → Anemic / Non-Anemic"
            active={state.mode === 'quick'}
            onPress={() => actions.setMode('quick')}
          />
          <View style={{ height: spacing.sm }} />
          <ModePill
            label="Full Screening"
            icon="clipboard-outline"
            description="Image + clinical data → Severity grade"
            active={state.mode === 'full'}
            onPress={() => actions.setMode('full')}
          />
        </SectionCard>

        {/* Step 2 — Patient info */}
        <SectionCard>
          <StepHeader
            step={2}
            title="Patient Information"
            subtitle="Optional — stored for records"
          />
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
        </SectionCard>

        {/* Step 3 — Clinical data (full mode only) */}
        {state.mode === 'full' && (
          <SectionCard>
            <StepHeader
              step={3}
              title="Clinical Data"
              subtitle="Required for severity classification"
              complete={derived.hasAge}
            />
            <InputField
              label={t.screening.patientAge}
              value={state.age}
              onChangeText={actions.setAge}
              placeholder={t.screening.agePlaceholder}
              keyboardType="numeric"
            />
            <GenderToggle value={state.gender} onChange={actions.setGender} />
            <InputField
              label={`${t.screening.hemoglobin} (optional)`}
              value={state.hbLevel}
              onChangeText={actions.setHbLevel}
              placeholder={t.screening.hbPlaceholder}
              keyboardType="decimal-pad"
            />
          </SectionCard>
        )}

        {/* Step 4 — Image */}
        <SectionCard>
          <StepHeader
            step={state.mode === 'full' ? 4 : 3}
            title="Conjunctiva Image"
            subtitle="Inner eyelid photograph"
            complete={derived.hasImage}
          />

          {derived.hasImage ? (
            <View style={styles.imageContainer}>
              <Image source={{ uri: derived.imageUri! }} style={styles.imagePreview} />
              <TouchableOpacity
                style={styles.changeImageBtn}
                onPress={() => router.push('/image-capture')}
                activeOpacity={0.85}
              >
                <Ionicons name="camera-outline" size={14} color={colors.white} />
                <Text style={styles.changeImageText}>Retake</Text>
              </TouchableOpacity>
              <View style={styles.imageReadyBadge}>
                <Ionicons name="checkmark-circle" size={14} color={colors.success} />
                <Text style={styles.imageReadyText}>Image ready</Text>
              </View>
            </View>
          ) : (
            <TouchableOpacity
              style={styles.cameraPlaceholder}
              onPress={() => router.push('/image-capture')}
              activeOpacity={0.85}
            >
              <View style={styles.cameraIconWrap}>
                <Ionicons name="camera" size={30} color={colors.primary} />
              </View>
              <Text style={styles.cameraTitle}>Capture Conjunctiva</Text>
              <Text style={styles.cameraHint}>
                Gently pull the lower eyelid down and photograph the inner surface
              </Text>
              <View style={styles.cameraCta}>
                <Ionicons name="add" size={14} color={colors.white} />
                <Text style={styles.cameraCtaText}>Open Camera</Text>
              </View>
            </TouchableOpacity>
          )}
        </SectionCard>

        {/* Offline notice */}
        {state.offlineQueued && (
          <View style={styles.noticeBox}>
            <Ionicons name="cloud-outline" size={18} color={colors.primary} />
            <Text style={styles.noticeText}>
              You're offline. Request queued — results will appear when you reconnect.
            </Text>
          </View>
        )}

        {/* Error */}
        {derived.error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle-outline" size={18} color={colors.error} />
            <Text style={styles.errorText}>{derived.error}</Text>
          </View>
        )}
      </ScrollView>

      {/* ── Fixed bottom action bar ─────────────────────────────────────────── */}
      <View style={[styles.actionBar, { paddingBottom: insets.bottom + spacing.sm }]}>
        <View style={styles.actionInner}>
          <View style={{ flex: 1 }}>
            <Text style={styles.actionTitle}>
              {state.mode === 'quick' ? 'Quick Screen' : 'Full Screening'}
            </Text>
            <Text style={styles.actionSub}>
              {derived.canSubmit ? 'Ready to analyse' : 'Complete required fields above'}
            </Text>
          </View>
          <TouchableOpacity
            style={[styles.analyseBtn, !derived.canSubmit && styles.analyseBtnDisabled]}
            onPress={actions.handleSubmit}
            disabled={!derived.canSubmit || derived.isLoading}
            activeOpacity={0.85}
          >
            <Ionicons
              name="pulse-outline"
              size={18}
              color={derived.canSubmit ? colors.white : colors.disabled}
            />
            <Text style={[styles.analyseBtnText, !derived.canSubmit && styles.analyseBtnTextDisabled]}>
              Analyse
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      <LoadingOverlay
        visible={derived.isLoading}
        message={state.mode === 'quick' ? 'Analysing conjunctiva…' : 'Analysing patient data…'}
      />
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const cardShadow = Platform.select({
  ios:     { shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 8, shadowOffset: { width: 0, height: 2 } },
  android: { elevation: 2 },
  default: {},
});

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: colors.background },
  scroll:  { flex: 1 },
  content: { padding: spacing.md, gap: spacing.md },

  // Progress
  progressRow:  { marginBottom: spacing.xs },
  progressLabel: { ...typography.caption, color: colors.textSecondary, marginBottom: 6 },
  progressTrack: {
    height: 4, backgroundColor: colors.borderLight,
    borderRadius: borderRadius.full, overflow: 'hidden',
  },
  progressFill: {
    height: '100%', backgroundColor: colors.primary,
    borderRadius: borderRadius.full,
  },

  // Card
  sectionCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.borderLight,
    ...cardShadow,
  },

  // Step header
  stepHeader: {
    flexDirection: 'row', alignItems: 'center',
    gap: spacing.sm, marginBottom: spacing.md,
  },
  stepBadge: {
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: colors.primary,
    alignItems: 'center', justifyContent: 'center',
  },
  stepBadgeComplete: { backgroundColor: colors.success },
  stepNumber: { fontSize: 13, fontWeight: '700', color: colors.white },
  stepTitle:    { ...typography.bodyBold, color: colors.text },
  stepSubtitle: { ...typography.caption, color: colors.textSecondary, marginTop: 1 },

  // Mode pills
  modePill: {
    flexDirection: 'row', alignItems: 'center',
    padding: spacing.md, borderRadius: borderRadius.md,
    borderWidth: 1.5, borderColor: colors.border,
    backgroundColor: colors.background,
  },
  modePillActive:      { backgroundColor: colors.primary, borderColor: colors.primary },
  modePillLabel:       { ...typography.bodyBold, color: colors.primary },
  modePillLabelActive: { color: colors.white },
  modePillDesc:        { ...typography.caption, color: colors.textSecondary, marginTop: 1 },
  modePillDescActive:  { color: 'rgba(255,255,255,0.75)' },

  // Image
  imageContainer: { borderRadius: borderRadius.md, overflow: 'hidden' },
  imagePreview:   { width: '100%', height: 210, borderRadius: borderRadius.md },
  changeImageBtn: {
    position: 'absolute', top: spacing.sm, right: spacing.sm,
    flexDirection: 'row', alignItems: 'center', gap: 4,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: borderRadius.full,
  },
  changeImageText: { ...typography.caption, color: colors.white, fontWeight: '600' },
  imageReadyBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: spacing.sm,
  },
  imageReadyText: { ...typography.caption, color: colors.success, fontWeight: '600' },

  // Camera placeholder
  cameraPlaceholder: {
    alignItems: 'center', padding: spacing.xl,
    borderRadius: borderRadius.md, borderWidth: 1.5,
    borderStyle: 'dashed', borderColor: colors.border,
    backgroundColor: colors.surfaceElevated, gap: spacing.sm,
  },
  cameraIconWrap: {
    width: 60, height: 60, borderRadius: 30,
    backgroundColor: colors.primaryLight,
    alignItems: 'center', justifyContent: 'center', marginBottom: spacing.xs,
  },
  cameraTitle: { ...typography.bodyBold, color: colors.text },
  cameraHint: {
    ...typography.caption, color: colors.textSecondary,
    textAlign: 'center', lineHeight: 18,
  },
  cameraCta: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg, paddingVertical: spacing.sm,
    borderRadius: borderRadius.full, marginTop: spacing.xs,
  },
  cameraCtaText: { ...typography.captionBold, color: colors.white },

  // Notice / error
  noticeBox: {
    flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm,
    backgroundColor: colors.surfaceElevated, borderRadius: borderRadius.md,
    borderWidth: 1, borderColor: colors.border, padding: spacing.md,
  },
  noticeText: { ...typography.caption, color: colors.textSecondary, flex: 1 },
  errorBox: {
    flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm,
    backgroundColor: colors.errorBg, borderRadius: borderRadius.md, padding: spacing.md,
  },
  errorText: { ...typography.caption, color: colors.error, flex: 1 },

  // Action bar
  actionBar: {
    backgroundColor: colors.surface,
    borderTopWidth: 1, borderTopColor: colors.borderLight,
    paddingTop: spacing.md, paddingHorizontal: spacing.md,
    ...Platform.select({
      ios:     { shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 12, shadowOffset: { width: 0, height: -3 } },
      android: { elevation: 10 },
      default: {},
    }),
  },
  actionInner: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  actionTitle: { ...typography.bodyBold, color: colors.text },
  actionSub:   { ...typography.caption, color: colors.textSecondary, marginTop: 1 },
  analyseBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg, paddingVertical: 14,
    borderRadius: borderRadius.full,
  },
  analyseBtnDisabled:     { backgroundColor: colors.surfaceElevated },
  analyseBtnText:         { ...typography.button, color: colors.white, fontSize: 15 },
  analyseBtnTextDisabled: { color: colors.disabled },
});
