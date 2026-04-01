/**
 * app/image-capture.tsx — Image picker with preview and confirm
 */
import React, { useState, useCallback } from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius, shadows } from '../src/shared/theme';
import { Button, Card } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import { usePlatformCamera } from '../src/shared/hooks/usePlatformCamera';
import { useAnalyticsStore } from '../src/store/analyticsStore';

export default function ImageCaptureScreen() {
  const router = useRouter();
  const [localUri, setLocalUri] = useState<string | null>(null);
  const setImageUri = useStore((s) => s.setImageUri);
  const { pickImage, takePhoto } = usePlatformCamera();
  const trackEvent = useAnalyticsStore((s) => s.trackEvent);

  const handlePickImage = useCallback(async () => {
    const uri = await pickImage();
    if (uri) {
      setLocalUri(uri);
    }
  }, [pickImage]);

  const handleTakePhoto = useCallback(async () => {
    const uri = await takePhoto();
    if (uri) {
      setLocalUri(uri);
    }
  }, [takePhoto]);

  const handleConfirm = useCallback(() => {
    if (localUri) {
      setImageUri(localUri);
      trackEvent('image_captured');
      router.back();
    }
  }, [localUri, router, setImageUri, trackEvent]);

  const handleRetake = useCallback(() => {
    setLocalUri(null);
  }, []);

  return (
    <View style={styles.container}>
      {localUri ? (
        // ── Preview Mode ──
        <View style={styles.previewContainer}>
          <Card variant="elevated" style={styles.previewCard}>
            <Image source={{ uri: localUri }} style={styles.previewImage} />
          </Card>

          <Text style={styles.confirmText}>
            Is this image clear enough for analysis?
          </Text>

          <View style={styles.buttonRow}>
            <Button
              title="Retake"
              onPress={handleRetake}
              variant="outline"
              style={styles.halfButton}
              icon={<Ionicons name="refresh" size={18} color={colors.primary} />}
            />
            <View style={{ width: spacing.md }} />
            <Button
              title="Confirm"
              onPress={handleConfirm}
              variant="primary"
              style={styles.halfButton}
              icon={<Ionicons name="checkmark" size={18} color={colors.white} />}
            />
          </View>
        </View>
      ) : (
        // ── Pick Mode ──
        <View style={styles.pickContainer}>
          <View style={styles.illustrationContainer}>
            <View style={styles.illustrationCircle}>
              <Ionicons name="eye-outline" size={64} color={colors.primary} />
            </View>
          </View>

          <Text style={styles.title}>Conjunctiva Image</Text>
          <Text style={styles.description}>
            Select a clear photo of the inner eyelid (conjunctiva).{'\n'}
            This helps our AI model assess anemia severity.
          </Text>

          <View style={styles.tipsCard}>
            <Text style={styles.tipsTitle}>Tips for best results:</Text>
            {[
              'Use good, natural lighting',
              'Focus on the inner lower eyelid',
              'Avoid blurry or dark images',
              'Remove any makeup near the eye area',
            ].map((tip, i) => (
              <View key={i} style={styles.tipRow}>
                <Ionicons name="checkmark-circle" size={16} color={colors.accent} />
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </View>

          <View style={styles.buttonRow}>
            <Button
              title="Camera"
              onPress={handleTakePhoto}
              variant="primary"
              style={styles.halfButton}
              icon={<Ionicons name="camera-outline" size={20} color={colors.white} />}
            />
            <View style={{ width: spacing.md }} />
            <Button
              title="Gallery"
              onPress={handlePickImage}
              variant="outline"
              style={styles.halfButton}
              icon={<Ionicons name="images-outline" size={20} color={colors.primary} />}
            />
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.lg,
  },
  // ── Pick Mode ──
  pickContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  illustrationContainer: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  illustrationCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    ...typography.title,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  description: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
    lineHeight: 22,
  },
  tipsCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.xl,
  },
  tipsTitle: {
    ...typography.captionBold,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  tipText: {
    ...typography.caption,
    color: colors.textSecondary,
    marginLeft: spacing.sm,
  },
  // ── Preview Mode ──
  previewContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  previewCard: {
    padding: 0,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: 300,
    borderRadius: borderRadius.lg,
  },
  confirmText: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginVertical: spacing.lg,
  },
  buttonRow: {
    flexDirection: 'row',
  },
  halfButton: {
    flex: 1,
  },
});
