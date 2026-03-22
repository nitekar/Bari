/**
 * app/index.tsx — Home screen (welcome + CTAs)
 */
import React from 'react';
import { View, Text, StyleSheet, StatusBar } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius, shadows } from '../src/shared/theme';
import { Button } from '../src/shared/components';
import { useStore } from '../src/app/store/useStore';

export default function HomeScreen() {
  const router = useRouter();
  const reset = useStore((s) => s.reset);

  const handleStartScreening = () => {
    reset();
    router.push('/screening');
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.background} />

      {/* Hero Section */}
      <View style={styles.hero}>
        <View style={styles.iconContainer}>
          <View style={styles.iconCircleOuter}>
            <View style={styles.iconCircleInner}>
              <Ionicons name="eye-outline" size={48} color={colors.white} />
            </View>
          </View>
        </View>

        <Text style={styles.title}>Anemia{'\n'}Screening</Text>
        <Text style={styles.subtitle}>
          AI-powered conjunctiva analysis for{'\n'}early anemia detection
        </Text>
      </View>

      {/* Info Cards */}
      <View style={styles.infoRow}>
        <View style={[styles.infoCard, { backgroundColor: colors.severityNormalBg }]}>
          <Ionicons name="camera-outline" size={24} color={colors.severityNormal} />
          <Text style={styles.infoTitle}>Image Analysis</Text>
          <Text style={styles.infoText}>Conjunctiva photo</Text>
        </View>
        <View style={[styles.infoCard, { backgroundColor: colors.severityMildBg }]}>
          <Ionicons name="analytics-outline" size={24} color={colors.severityModerate} />
          <Text style={styles.infoTitle}>AI Prediction</Text>
          <Text style={styles.infoText}>Severity & confidence</Text>
        </View>
        <View style={[styles.infoCard, { backgroundColor: '#EDE7F6' }]}>
          <Ionicons name="nutrition-outline" size={24} color="#7E57C2" />
          <Text style={styles.infoTitle}>Guidance</Text>
          <Text style={styles.infoText}>Nutrition advice</Text>
        </View>
      </View>

      {/* CTAs */}
      <View style={styles.ctaContainer}>
        <Button
          title="Start Screening"
          onPress={handleStartScreening}
          variant="primary"
          icon={<Ionicons name="medkit-outline" size={20} color={colors.white} />}
        />
        <View style={{ height: spacing.md }} />
        <Button
          title="Learn About Anemia"
          onPress={() => router.push('/education')}
          variant="outline"
          icon={<Ionicons name="book-outline" size={20} color={colors.primary} />}
        />
        <View style={{ height: spacing.sm }} />
        <Button
          title="View Analytics"
          onPress={() => router.push('/analytics')}
          variant="secondary"
          icon={<Ionicons name="bar-chart-outline" size={20} color={colors.text} />}
        />
      </View>

      {/* Version */}
      <Text style={styles.version}>v2.0.0 • Bari Capstone</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: spacing.lg,
    paddingTop: 60,
  },
  hero: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  iconContainer: {
    marginBottom: spacing.lg,
  },
  iconCircleOuter: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: colors.accent + '40',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconCircleInner: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.lg,
  },
  title: {
    ...typography.heroTitle,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
  },
  infoCard: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.md,
    marginHorizontal: 4,
  },
  infoTitle: {
    ...typography.captionBold,
    color: colors.text,
    marginTop: spacing.sm,
    textAlign: 'center',
  },
  infoText: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: 2,
  },
  ctaContainer: {
    marginTop: 'auto',
    marginBottom: spacing.lg,
  },
  version: {
    ...typography.caption,
    color: colors.textLight,
    textAlign: 'center',
    marginBottom: spacing.md,
  },
});
