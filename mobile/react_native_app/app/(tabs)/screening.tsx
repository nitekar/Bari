/**
 * app/screening.tsx — Main screening form
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Image,
  Alert,
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
import { predictMultimodal } from '../../src/services/screeningService';
import type { ScreeningResult } from '../../src/services/screeningService';
import { useAnalyticsStore } from '../../src/store/analyticsStore';

export default function ScreeningScreen() {
  const router = useRouter();
  const { t } = useTranslation();

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

  // ── Submit handler — always multimodal ──
  const handleSubmit = useCallback(async () => {
    if (!hasAge) {
      Alert.alert('Missing Data', 'Please enter the patient age.');
      return;
    }
    if (!hasImage) {
      Alert.alert('Missing Data', 'Please add a conjunctiva image to proceed.');
      return;
    }

    setLoading(true);
    setError(null);
    trackEvent('screening_started', { mode: 'multimodal' });

    try {
      const result = await predictMultimodal(
        { imageUri: imageUri!, age: ageNum, gender, hb_level: hbNum },
        userId,
      );

      setResult(result);
      setLoading(false);

      trackEvent('screening_completed', { severity: result.prediction, confidence: result.confidence });
      addToHistory({
        id: Date.now().toString(),
        date: new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        mode: 'multimodal',
        age: ageNum,
        gender,
        imageUrl: result.imageStoragePath,
        patientName: patientName.trim() || undefined,
        patientLocation: patientLocation.trim() || undefined,
      });

      router.push('/result');
    } catch (err: any) {
      setError(err.message || 'Screening failed. Please try again.');
    }
  }, [age, gender, hbLevel, imageUri, patientName, patientLocation]);

  return (
    <View style={styles.wrapper}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {/* Patient Information Section */}
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
            title="Run Screening"
            onPress={handleSubmit}
            variant="primary"
            loading={isLoading}
            disabled={!hasAge && !hasImage}
            icon={
              !isLoading ? (
                <Ionicons name="pulse-outline" size={20} color={colors.white} />
              ) : undefined
            }
          />
        </View>
      </ScrollView>

      <LoadingOverlay visible={isLoading} message="Analyzing patient data…" />
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
  modeCard: {
    backgroundColor: colors.primary + '15',
    marginBottom: spacing.lg,
  },
  modeRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  modeText: {
    ...typography.caption,
    color: colors.textSecondary,
    marginLeft: spacing.sm,
  },
  modeBold: {
    fontWeight: '700',
    color: colors.primary,
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
