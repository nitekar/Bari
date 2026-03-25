/**
 * app/screening.tsx — Main screening form
 * Supports two modes:
 *   Quick Screen — image only → binary (Anemic / Non-Anemic)
 *   Full Screen  — image + clinical data → 4-class severity
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Image,
  Alert,
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
import { useStore } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';
import { predictImage, predictMultimodal } from '../../src/services/screeningService';
import { useAnalyticsStore } from '../../src/store/analyticsStore';

type ScreeningMode = 'quick' | 'full';

export default function ScreeningScreen() {
  const router = useRouter();
  const { t } = useTranslation();

  // ── Mode ──
  const [mode, setMode] = useState<ScreeningMode>('full');

  // ── Local form state ──
  const [patientName, setPatientName] = useState('');
  const [patientLocation, setPatientLocation] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState(0);
  const [hbLevel, setHbLevel] = useState('');

  // ── Global state ──
  const imageUri = useStore((s) => s.imageUri);
  const userId = useStore((s) => s.userId);
  const isLoading = useStore((s) => s.isLoading);
  const error = useStore((s) => s.error);
  const setResult = useStore((s) => s.setResult);
  const setLoading = useStore((s) => s.setLoading);
  const setError = useStore((s) => s.setError);
  const addToHistory = useStore((s) => s.addToHistory);
  const trackEvent = useAnalyticsStore((s) => s.trackEvent);

  // ── Validation ──
  const ageNum = parseFloat(age);
  const hbNum = hbLevel ? parseFloat(hbLevel) : null;
  const hasAge = !isNaN(ageNum) && ageNum >= 0;
  const hasImage = !!imageUri;

  const canSubmit = mode === 'quick' ? hasImage : hasAge && hasImage;

  // ── Submit ──
  const handleSubmit = useCallback(async () => {
    if (!hasImage) {
      Alert.alert('Missing Image', 'Please add a conjunctiva image to proceed.');
      return;
    }
    if (mode === 'full' && !hasAge) {
      Alert.alert('Missing Data', 'Please enter the patient age.');
      return;
    }

    setLoading(true);
    setError(null);
    trackEvent('screening_started', { mode });

    try {
      const result = mode === 'quick'
        ? await predictImage(imageUri!, userId)
        : await predictMultimodal(
            { imageUri: imageUri!, age: ageNum, gender, hb_level: hbNum },
            userId,
          );

      setResult(result);
      setLoading(false);

      trackEvent('screening_completed', { severity: result.prediction, confidence: result.confidence, mode });
      addToHistory({
        id: Date.now().toString(),
        date: new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        mode: mode === 'quick' ? 'image' : 'multimodal',
        age: mode === 'full' ? ageNum : undefined,
        gender: mode === 'full' ? gender : undefined,
        imageUrl: result.imageStoragePath,
        patientName: patientName.trim() || undefined,
        patientLocation: patientLocation.trim() || undefined,
      });

      router.push('/result');
    } catch (err: any) {
      setError(err.message || 'Screening failed. Please try again.');
    }
  }, [mode, age, gender, hbLevel, imageUri, patientName, patientLocation]);

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
            style={[styles.modeBtn, mode === 'quick' && styles.modeBtnActive]}
            onPress={() => setMode('quick')}
            activeOpacity={0.8}
          >
            <Ionicons
              name="eye-outline"
              size={16}
              color={mode === 'quick' ? colors.white : colors.primary}
            />
            <Text style={[styles.modeBtnText, mode === 'quick' && styles.modeBtnTextActive]}>
              Quick Screen
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.modeBtn, mode === 'full' && styles.modeBtnActive]}
            onPress={() => setMode('full')}
            activeOpacity={0.8}
          >
            <Ionicons
              name="clipboard-outline"
              size={16}
              color={mode === 'full' ? colors.white : colors.primary}
            />
            <Text style={[styles.modeBtnText, mode === 'full' && styles.modeBtnTextActive]}>
              Full Screen
            </Text>
          </TouchableOpacity>
        </View>

        {/* Mode description */}
        <Text style={styles.modeHint}>
          {mode === 'quick'
            ? 'Image only — detects Anemic / Non-Anemic'
            : 'Image + clinical data — classifies severity (Mild / Moderate / Severe)'}
        </Text>

        {/* Patient Info — always shown */}
        <Text style={styles.sectionTitle}>{t.screening.patientInfo}</Text>

        <InputField
          label={t.screening.patientName}
          value={patientName}
          onChangeText={setPatientName}
          placeholder="e.g. Amara Uwase"
        />

        <InputField
          label={t.screening.location}
          value={patientLocation}
          onChangeText={setPatientLocation}
          placeholder="e.g. Kigali, Rwanda"
        />

        {/* Clinical fields — Full Screen only */}
        {mode === 'full' && (
          <>
            <InputField
              label={t.screening.patientAge}
              value={age}
              onChangeText={setAge}
              placeholder={t.screening.agePlaceholder}
              keyboardType="numeric"
            />

            <GenderToggle value={gender} onChange={setGender} />

            <InputField
              label={`${t.screening.hemoglobin} — ${t.screening.optional}`}
              value={hbLevel}
              onChangeText={setHbLevel}
              placeholder={t.screening.hbPlaceholder}
              keyboardType="decimal-pad"
            />
          </>
        )}

        {/* Image Section */}
        <Text style={styles.sectionTitle}>Conjunctiva Image</Text>

        {imageUri ? (
          <Card>
            <Image source={{ uri: imageUri }} style={styles.imagePreview} />
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
        {error && <ErrorMessage message={error} onRetry={handleSubmit} />}

        {/* Submit */}
        <View style={styles.submitContainer}>
          <Button
            title={mode === 'quick' ? 'Quick Screen' : 'Run Full Screening'}
            onPress={handleSubmit}
            variant="primary"
            loading={isLoading}
            disabled={!canSubmit}
            icon={
              !isLoading ? (
                <Ionicons name="pulse-outline" size={20} color={colors.white} />
              ) : undefined
            }
          />
        </View>
      </ScrollView>

      <LoadingOverlay
        visible={isLoading}
        message={mode === 'quick' ? 'Analyzing image…' : 'Analyzing patient data…'}
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
