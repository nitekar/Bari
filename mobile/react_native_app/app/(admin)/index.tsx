/**
 * app/(admin)/index.tsx — Admin Dashboard
 * Stats: total screenings, severe rate, active model. Recent activity list.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card, Button } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { signOut } from '../../src/services/supabaseAuth';

const MODEL_VERSION = 'BariNet v2.1';

export default function AdminDashboard() {
  const router = useRouter();
  const history = useStore((s) => s.history);
  const setUserId = useStore((s) => s.setUserId);

  const total = history.length;
  const severeCount = history.filter(r => r.prediction === 'Severe').length;
  const severeRate = total > 0 ? Math.round((severeCount / total) * 100) : 0;
  const today = new Date().toDateString();
  const todayCount = history.filter(r => new Date(r.date).toDateString() === today).length;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {/* Hero */}
      <View style={styles.hero}>
        <View style={styles.heroIconBox}>
          <Ionicons name="shield-checkmark-outline" size={28} color={colors.primary} />
        </View>
        <Text style={styles.heroTitle}>Admin Dashboard</Text>
        <Text style={styles.heroSub}>System overview and management</Text>
      </View>

      {/* KPI cards */}
      <View style={styles.grid}>
        <KpiCard icon="pulse-outline"    color={colors.primary}       label="Total Screenings" value={total.toString()} />
        <KpiCard icon="today-outline"    color={colors.secondary}     label="Today"            value={todayCount.toString()} />
        <KpiCard icon="warning-outline"  color={colors.error}         label="Severe Rate"      value={`${severeRate}%`} />
        <KpiCard icon="hardware-chip-outline" color="#7E57C2"         label="Active Model"     value={MODEL_VERSION} small />
      </View>

      {/* Recent activity */}
      {history.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          {history.slice(0, 10).map(r => (
            <Card key={r.id} style={styles.activityCard}>
              <View style={styles.activityRow}>
                <View style={[styles.dot, { backgroundColor: dotColor(r.prediction) }]} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.activityName}>{r.patientName ?? 'Anonymous'}</Text>
                  <Text style={styles.activitySub}>
                    {r.prediction} · {r.mode} · {new Date(r.date).toLocaleString()}
                  </Text>
                </View>
                <Text style={styles.conf}>{Math.round(r.confidence * 100)}%</Text>
              </View>
            </Card>
          ))}
        </>
      )}

      <View style={styles.logoutContainer}>
        <Button
          title="Sign Out"
          onPress={async () => {
            await signOut();
            setUserId(null);
          }}
          variant="outline"
          icon={<Ionicons name="log-out-outline" size={20} color={colors.error} />}
        />
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function KpiCard({ icon, color, label, value, small }: { icon: any; color: string; label: string; value: string; small?: boolean }) {
  return (
    <Card style={[styles.kpiCard, { borderTopColor: color, borderTopWidth: 3 }]}>
      <Ionicons name={icon} size={20} color={color} />
      <Text style={[styles.kpiValue, small && { fontSize: 13 }]}>{value}</Text>
      <Text style={styles.kpiLabel}>{label}</Text>
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
  hero: { alignItems: 'center', paddingVertical: spacing.xl, marginBottom: spacing.lg },
  heroIconBox: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.primary + '20', alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md },
  heroTitle: { fontSize: 24, fontWeight: '800', color: colors.text },
  heroSub: { fontSize: 13, color: colors.textSecondary, marginTop: 4 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginBottom: spacing.lg },
  kpiCard: { width: '47%', alignItems: 'center', paddingVertical: spacing.md },
  kpiValue: { fontSize: 22, fontWeight: '800', color: colors.text, marginTop: 4 },
  kpiLabel: { fontSize: 11, color: colors.textSecondary, textAlign: 'center' },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  activityCard: { paddingVertical: spacing.sm, marginBottom: spacing.sm },
  activityRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  dot: { width: 10, height: 10, borderRadius: 5 },
  activityName: { fontSize: 14, fontWeight: '600', color: colors.text },
  activitySub: { fontSize: 11, color: colors.textSecondary },
  conf: { fontSize: 12, color: colors.textLight },
  logoutContainer: { marginTop: spacing.xl, paddingHorizontal: spacing.xl },
});
