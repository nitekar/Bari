/**
 * app/(parent)/index.tsx — Parent Home Dashboard
 * Child name greeting, latest screening result, baby summary, quick actions.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter, Redirect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card, Button } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';

function riskLabel(prediction: string) {
  if (prediction === 'Severe' || prediction === 'Anemic') return { label: 'High Risk', color: colors.error };
  if (prediction === 'Moderate') return { label: 'Moderate Risk', color: '#e67e22' };
  if (prediction === 'Mild') return { label: 'Low Risk', color: '#f1c40f' };
  return { label: 'Healthy', color: colors.primary };
}

export default function ParentDashboard() {
  const router = useRouter();
  const { t } = useTranslation();
  const childName = useStore((s) => s.childName);
  const userId = useStore((s) => s.userId);

  if (!userId) {
    return <Redirect href="/education" />;
  }
  const history = useStore((s) => s.history);
  const sleepLogs = useStore((s) => s.sleepLogs);
  const feedingLogs = useStore((s) => s.feedingLogs);
  const childAgeMonths = useStore((s) => s.childAgeMonths);

  const latestScreening = history[0] ?? null;
  const lastSleep = sleepLogs[0] ?? null;
  const lastFeeding = feedingLogs[0] ?? null;
  const risk = latestScreening ? riskLabel(latestScreening.prediction) : null;

  const greeting = childName ? `${t.parentDashboard.hi}, ${childName}!` : t.parentDashboard.welcome;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {/* Hero */}
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="heart" size={28} color={colors.secondary} />
        </View>
        <Text style={styles.heroTitle}>{greeting}</Text>
        <Text style={styles.heroSub}>
          {childAgeMonths != null ? `${childAgeMonths} ${t.parentDashboard.monthsOld}` : t.parentDashboard.heroSub}
        </Text>
      </View>

      {/* Latest screening result */}
      {latestScreening && risk && (
        <>
          <Text style={styles.sectionTitle}>{t.history.lastScreening}</Text>
          <Card style={[styles.resultCard, { borderLeftColor: risk.color, borderLeftWidth: 4 }]}>
            <View style={styles.resultRow}>
              <View style={[styles.riskBadge, { backgroundColor: risk.color + '20' }]}>
                <Text style={[styles.riskLabel, { color: risk.color }]}>{risk.label}</Text>
              </View>
              <Text style={styles.resultDate}>{new Date(latestScreening.date).toLocaleDateString()}</Text>
            </View>
            <Text style={styles.resultAction}>
              {latestScreening.prediction === 'Severe'
                ? t.parentDashboard.severeAlert
                : latestScreening.prediction === 'Moderate'
                ? t.parentDashboard.moderateAlert
                : t.parentDashboard.healthyMessage}
            </Text>
          </Card>
        </>
      )}

      {/* Baby summary */}
      <Text style={styles.sectionTitle}>{t.parentDashboard.babySummary}</Text>
      <View style={styles.summaryRow}>
        <TouchableOpacity style={styles.summaryCard} onPress={() => router.push('/parent-sleep')} activeOpacity={0.8}>
          <Ionicons name="moon-outline" size={24} color={colors.primaryDark} />
          <Text style={styles.summaryCardTitle}>{t.parentDashboard.sleep}</Text>
          <Text style={styles.summaryCardSub}>
            {lastSleep ? `${lastSleep.durationHours.toFixed(1)}h on ${new Date(lastSleep.date).toLocaleDateString()}` : t.parentDashboard.noEntries}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.summaryCard} onPress={() => router.push('/parent-feeding')} activeOpacity={0.8}>
          <Ionicons name="nutrition-outline" size={24} color={colors.secondary} />
          <Text style={styles.summaryCardTitle}>{t.parentDashboard.feeding}</Text>
          <Text style={styles.summaryCardSub}>
            {lastFeeding ? `${lastFeeding.type} at ${lastFeeding.time}` : t.parentDashboard.noEntries}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.summaryCard} onPress={() => router.push('/parent-development')} activeOpacity={0.8}>
          <Ionicons name="trending-up-outline" size={24} color={colors.accentDark} />
          <Text style={styles.summaryCardTitle}>{t.parentDashboard.growth}</Text>
          <Text style={styles.summaryCardSub}>
            {childAgeMonths != null ? `${childAgeMonths} mo` : t.parentDashboard.setAge}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Quick actions */}
      <Text style={styles.sectionTitle}>{t.parentDashboard.quickActions}</Text>
      <Button
        title={t.home.viewHistory}
        onPress={() => router.push('/(parent)/results')}
        variant="primary"
        icon={<Ionicons name="pulse-outline" size={20} color={colors.white} />}
      />
      <View style={{ height: spacing.sm }} />
      <Button
        title={t.result.learnMore}
        onPress={() => router.push('/education')}
        variant="secondary"
        icon={<Ionicons name="book-outline" size={20} color={colors.text} />}
      />

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  hero: { alignItems: 'center', paddingVertical: spacing.xl, marginBottom: spacing.md },
  heroIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.secondary + '20', alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md },
  heroTitle: { fontSize: 26, fontWeight: '800', color: colors.text },
  heroSub: { fontSize: 14, color: colors.textSecondary, marginTop: 4 },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  resultCard: { marginBottom: spacing.lg, paddingVertical: spacing.md },
  resultRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.sm },
  riskBadge: { paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: borderRadius.pill },
  riskLabel: { fontSize: 12, fontWeight: '700' },
  resultDate: { fontSize: 11, color: colors.textLight },
  resultAction: { fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
  summaryRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.lg },
  summaryCard: {
    flex: 1, backgroundColor: colors.surface, borderRadius: borderRadius.lg,
    padding: spacing.md, alignItems: 'center', ...shadows.sm,
  },
  summaryCardTitle: { fontSize: 12, fontWeight: '700', color: colors.text, marginTop: spacing.xs },
  summaryCardSub: { fontSize: 10, color: colors.textSecondary, textAlign: 'center', marginTop: 2 },
});
