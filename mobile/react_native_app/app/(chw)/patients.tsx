/**
 * app/(chw)/patients.tsx — Patient list grouped by name
 */
import React, { useMemo } from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';

interface PatientSummary {
  name: string;
  lastDate: string;
  lastPrediction: string;
  screeningCount: number;
}

function severityColor(prediction: string): string {
  if (prediction === 'Severe') return colors.error;
  if (prediction === 'Moderate') return '#e67e22';
  if (prediction === 'Mild') return '#f1c40f';
  return colors.primary;
}

function severityBg(prediction: string): string {
  if (prediction === 'Severe') return colors.error + '18';
  if (prediction === 'Moderate') return '#e67e2218';
  if (prediction === 'Mild') return '#f1c40f18';
  return colors.primary + '18';
}

export default function PatientsScreen() {
  const history = useStore((s) => s.history);

  const patients: PatientSummary[] = useMemo(() => {
    const map = new Map<string, PatientSummary>();
    for (const record of history) {
      const name = record.patientName?.trim() || 'Unknown';
      const existing = map.get(name);
      if (!existing || new Date(record.date) > new Date(existing.lastDate)) {
        map.set(name, {
          name,
          lastDate: record.date,
          lastPrediction: record.prediction,
          screeningCount: (existing?.screeningCount ?? 0) + 1,
        });
      } else {
        map.set(name, { ...existing, screeningCount: existing.screeningCount + 1 });
      }
    }
    return Array.from(map.values()).sort(
      (a, b) => new Date(b.lastDate).getTime() - new Date(a.lastDate).getTime(),
    );
  }, [history]);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Patients</Text>
        <Text style={styles.count}>{patients.length} total</Text>
      </View>

      {patients.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="people-outline" size={48} color={colors.textLight} />
          <Text style={styles.emptyText}>No patients yet</Text>
          <Text style={styles.emptySub}>Complete a screening to add patients.</Text>
        </View>
      ) : (
        patients.map((p) => (
          <Card key={p.name} style={styles.card}>
            <View style={styles.row}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>
                  {p.name !== 'Unknown' ? p.name.slice(0, 2).toUpperCase() : '??'}
                </Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.name}>{p.name}</Text>
                <Text style={styles.sub}>
                  {p.screeningCount} screening{p.screeningCount !== 1 ? 's' : ''} · Last: {new Date(p.lastDate).toLocaleDateString()}
                </Text>
              </View>
              <View style={[styles.badge, { backgroundColor: severityBg(p.lastPrediction) }]}>
                <Text style={[styles.badgeText, { color: severityColor(p.lastPrediction) }]}>
                  {p.lastPrediction}
                </Text>
              </View>
            </View>
          </Card>
        ))
      )}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.lg },
  title: { fontSize: 22, fontWeight: '800', color: colors.text },
  count: { fontSize: 13, color: colors.textSecondary },
  empty: { alignItems: 'center', paddingVertical: spacing.xxl },
  emptyText: { fontSize: 16, fontWeight: '600', color: colors.textSecondary, marginTop: spacing.md },
  emptySub: { fontSize: 13, color: colors.textLight, marginTop: 4 },
  card: { marginBottom: spacing.sm, paddingVertical: spacing.sm },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { fontSize: 15, fontWeight: '800', color: colors.primaryDark },
  name: { fontSize: 15, fontWeight: '600', color: colors.text },
  sub: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: borderRadius.pill,
  },
  badgeText: { fontSize: 11, fontWeight: '700' },
});
