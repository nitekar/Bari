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
  Platform,
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
import {
  predictTabular,
  predictImage,
  predictMultimodal,
} from '../../src/services/screeningService';
import type { ScreeningResult } from '../../src/services/screeningService';
import { useAnalyticsStore } from '../../src/store/analyticsStore';

export default function ScreeningScreen() {
  const router = useRouter();

  // ── Local form state ──
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

  // ── Determine which prediction mode to use ──
  const getScreeningMode = (): 'tabular' | 'image' | 'multimodal' => {
    if (hasImage && hasAge) return 'multimodal';
    if (hasImage) return 'image';
    return 'tabular';
  };

  // ── Submit handler ──
  const handleSubmit = useCallback(async () => {
    const mode = getScreeningMode();

    if (mode === 'tabular' && !hasAge) {
      if (Platform.OS === 'web') {
        window.alert('Missing Data: Please enter the patient age.');
      } else {
        Alert.alert('Missing Data', 'Please enter the patient age.');
      }
      return;
    }

    setLoading(true);
    setError(null);
    trackEvent('screening_started', { mode });

    try {
      let result: ScreeningResult;

      switch (mode) {
        case 'tabular':
          result = await predictTabular({
            age: ageNum,
            gender,
            hb_level: hbNum,
          });
          break;

        case 'image':
          result = await predictImage(imageUri!, userId);
          break;

        case 'multimodal':
          result = await predictMultimodal(
            {
              imageUri: imageUri!,
              age: ageNum,
              gender,
              hb_level: hbNum,
            },
            userId,
          );
          break;
      }

      setResult(result);
      setLoading(false);

      // Track + persist
      trackEvent('screening_completed', { severity: result.prediction, confidence: result.confidence });
      addToHistory({
        id: Date.now().toString(),
        date: new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        mode,
        age: ageNum,
        gender,
        imageUrl: result.imageStoragePath,
      });

      router.push('/result');
    } catch (err: any) {
      setError(err.message || 'Screening failed. Please try again.');
    }
  }, [age, gender, hbLevel, imageUri]);

  const screeningMode = getScreeningMode();

  return (
    <View style={styles.wrapper}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {/* Mode Indicator */}
        <Card style={styles.modeCard}>
          <View style={styles.modeRow}>
            <Ionicons
              name={
                screeningMode === 'multimodal'
                  ? 'layers-outline'
                  : screeningMode === 'image'
                  ? 'camera-outline'
                  : 'document-text-outline'
              }
              size={20}
              color={colors.primary}
            />
            <Text style={styles.modeText}>
              Mode:{' '}
              <Text style={styles.modeBold}>
                {screeningMode === 'multimodal'
                  ? 'Multimodal (Image + Data)'
                  : screeningMode === 'image'
                  ? 'Image Only'
                  : 'Tabular Data Only'}
              </Text>
            </Text>
          </View>
        </Card>

        {/* Patient Information Section */}
        <Text style={styles.sectionTitle}>Patient Information</Text>

        <InputField
          label="Age (months)"
          value={age}
          onChangeText={setAge}
          placeholder="e.g. 24"
          keyboardType="numeric"
        />

        <GenderToggle value={gender} onChange={setGender} />

        <InputField
          label="Hemoglobin Level (g/dL) — Optional"
          value={hbLevel}
          onChangeText={setHbLevel}
          placeholder="e.g. 11.5"
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
