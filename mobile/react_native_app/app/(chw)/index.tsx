/**
 * app/(chw)/index.tsx — CHW Dashboard
 * Shows today's screenings, totals, severe alerts, quick actions, recent records.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card, Button } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';

export default function ChwDashboard() {
  const router = useRouter();
  const { t } = useTranslation();
  const history = useStore((s) => s.history);
  const today = new Date().toDateString();
  const todayScreenings = history.filter(r => new Date(r.date).toDateString() === today);
  const severeCount = history.filter(r => r.prediction === 'Severe').length;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {/* Hero */}
      <View style={styles.hero}>
        <Text style={styles.heroTitle}>{t.chwDashboard.title}</Text>
        <Text style={styles.heroSub}>{t.chwDashboard.subtitle}</Text>
      </View>

      {/* Stats row */}
      <View style={styles.statsRow}>
        <StatCard label={t.chwDashboard.today} value={todayScreenings.length.toString()} icon="medkit" color={colors.primary} />
        <StatCard label={t.chwDashboard.total} value={history.length.toString()} icon="people" color={colors.secondary} />
        <StatCard label={t.chwDashboard.severe} value={severeCount.toString()} icon="warning" color={colors.error} />
      </View>

      {/* Quick actions */}
      <Text style={styles.sectionTitle}>{t.chwDashboard.quickActions}</Text>
      <Button title={t.chwDashboard.newScreening} onPress={() => router.push('/(chw)/screening')} variant="primary" icon={<Ionicons name="add-circle-outline" size={20} color={colors.white} />} />
      <View style={{ height: spacing.md }} />
      <Button title={t.chwDashboard.viewPatients} onPress={() => router.push('/(chw)/patients')} variant="secondary" icon={<Ionicons name="people-outline" size={20} color={colors.text} />} />

      {/* Recent screenings */}
      {history.length > 0 && (
        <>
          <Text style={[styles.sectionTitle, { marginTop: spacing.lg }]}>{t.chwDashboard.recentScreenings}</Text>
          {history.slice(0, 5).map(r => (
            <Card key={r.id} style={styles.recordCard}>
              <View style={styles.recordRow}>
                <View style={[styles.dot, { backgroundColor: dotColor(r.prediction) }]} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.recordName}>{r.patientName ?? t.chwDashboard.unknown}</Text>
                  <Text style={styles.recordSub}>{new Date(r.date).toLocaleDateString()} · {r.prediction}</Text>
                </View>
                <Text style={styles.conf}>{Math.round(r.confidence * 100)}%</Text>
              </View>
            </Card>
          ))}
        </>
      )}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function StatCard({ label, value, icon, color }: { label: string; value: string; icon: any; color: string }) {
  return (
    <Card style={[styles.statCard, { borderTopColor: color, borderTopWidth: 3 }]}>
      <Ionicons name={icon} size={20} color={color} />
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </Card>
  );
}

function dotColor(p: string) {
  if (p === 'Severe') return colors.error;
  if (p === 'Moderate') return '#e67e22';
  if (p === 'Mild') return '#f1c40f';
  return colors.primary;
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  hero: { paddingVertical: spacing.xl, marginBottom: spacing.lg },
  heroTitle: { fontSize: 26, fontWeight: '800', color: colors.text },
  heroSub: { fontSize: 14, color: colors.textSecondary, marginTop: 4 },
  statsRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.lg },
  statCard: { flex: 1, alignItems: 'center', paddingVertical: spacing.md },
  statValue: { fontSize: 22, fontWeight: '800', color: colors.text, marginTop: 4 },
  statLabel: { fontSize: 11, color: colors.textSecondary },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  recordCard: { paddingVertical: spacing.sm, marginBottom: spacing.sm },
  recordRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  dot: { width: 10, height: 10, borderRadius: 5 },
  recordName: { fontSize: 14, fontWeight: '600', color: colors.text },
  recordSub: { fontSize: 12, color: colors.textSecondary },
  conf: { fontSize: 12, color: colors.textLight },
});
