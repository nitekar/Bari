/**
 * app/(tabs)/index.tsx — Home screen
 */
import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Button, Logo } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';

export default function HomeScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const reset = useStore((s) => s.reset);

  const handleStartScreening = () => {
    reset();
    router.push('/screening');
  };

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.container}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Hero ── */}
      <View style={styles.hero}>
        {/* Cloud bubbles for baby theme */}
        <View style={styles.bubble1} />
        <View style={styles.bubble2} />

        <Logo size="lg" horizontal={false} showText />

        <Text style={styles.subtitle}>{t.home.subtitle}</Text>
      </View>

      {/* ── Info cards ── */}
      <View style={styles.cardRow}>
        <InfoCard
          icon="camera-outline"
          color={colors.severityNormal}
          bg={colors.severityNormalBg}
          title={t.home.imageAnalysis}
          sub={t.home.conjunctivaPhoto}
        />
        <InfoCard
          icon="analytics-outline"
          color={colors.severityModerate}
          bg={colors.severityMildBg}
          title={t.home.aiPrediction}
          sub={t.home.severityConfidence}
        />
        <InfoCard
          icon="nutrition-outline"
          color="#9575CD"
          bg="#EDE7F6"
          title={t.home.guidance}
          sub={t.home.nutritionAdvice}
        />
      </View>

      {/* ── CTAs ── */}
      <View style={styles.ctaSection}>
        <Button
          title={t.home.startScreening}
          onPress={handleStartScreening}
          variant="primary"
          icon={<Ionicons name="medkit-outline" size={20} color={colors.white} />}
        />
        <View style={{ height: spacing.md }} />
        <Button
          title={t.home.learnMore}
          onPress={() => router.push('/education')}
          variant="outline"
          icon={<Ionicons name="book-outline" size={20} color={colors.primary} />}
        />
        <View style={{ height: spacing.sm }} />
        <Button
          title={t.home.viewHistory}
          onPress={() => router.push('/history')}
          variant="secondary"
          icon={<Ionicons name="time-outline" size={20} color={colors.text} />}
        />
      </View>

      {/* ── Bottom padding for floating tab bar ── */}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function InfoCard({
  icon,
  color,
  bg,
  title,
  sub,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  bg: string;
  title: string;
  sub: string;
}) {
  return (
    <View style={[styles.infoCard, { backgroundColor: bg }]}>
      <View style={[styles.infoIconCircle, { backgroundColor: color + '30' }]}>
        <Ionicons name={icon} size={22} color={color} />
      </View>
      <Text style={styles.infoTitle}>{title}</Text>
      <Text style={styles.infoSub}>{sub}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: colors.background,
  },
  container: {
    paddingHorizontal: spacing.lg,
    paddingTop: 60,
  },
  // ── Hero ──
  hero: {
    alignItems: 'center',
    marginBottom: spacing.xl,
    position: 'relative',
    paddingTop: 24,
    paddingBottom: 32,
  },
  bubble1: {
    position: 'absolute',
    top: 0,
    right: -20,
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: colors.secondary + '40',
  },
  bubble2: {
    position: 'absolute',
    top: 20,
    left: -30,
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: colors.accent + '50',
  },
  subtitle: {
    fontSize: 15,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginTop: 14,
  },
  // ── Cards ──
  cardRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
    gap: 8,
  },
  infoCard: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: 6,
    borderRadius: borderRadius.lg,
    ...shadows.sm,
  },
  infoIconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  infoTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.text,
    textAlign: 'center',
  },
  infoSub: {
    fontSize: 10,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: 2,
  },
  // ── CTAs ──
  ctaSection: {
    marginTop: 'auto',
  },
});
