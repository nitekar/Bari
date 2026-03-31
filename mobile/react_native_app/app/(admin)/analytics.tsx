/**
 * app/(admin)/analytics.tsx — Severity distribution bars
 */
import React, { useMemo } from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';

const SEVERITY_ORDER = ['Non-Anemic', 'Mild', 'Moderate', 'Severe', 'Anemic'];
const SEVERITY_COLORS: Record<string, string> = {
  'Non-Anemic': colors.primary,
  Mild: '#f1c40f',
  Moderate: '#e67e22',
  Severe: colors.error,
  Anemic: '#e74c3c',
};

export default function AnalyticsScreen() {
  const history = useStore((s) => s.history);

  const distribution = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const r of history) {
      counts[r.prediction] = (counts[r.prediction] ?? 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => {
      const ai = SEVERITY_ORDER.indexOf(a[0]);
      const bi = SEVERITY_ORDER.indexOf(b[0]);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
  }, [history]);

  const total = history.length;

  // Mode breakdown
  const modeCount = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const r of history) {
      counts[r.mode] = (counts[r.mode] ?? 0) + 1;
    }
    return counts;
  }, [history]);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.pageTitle}>Analytics</Text>

      {/* Summary */}
      <Card style={styles.summaryCard}>
        <View style={styles.summaryRow}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>{total}</Text>
            <Text style={styles.summaryLabel}>Total Screenings</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>{distribution.length}</Text>
            <Text style={styles.summaryLabel}>Unique Outcomes</Text>
          </View>
        </View>
      </Card>

      {/* Severity distribution */}
      <Text style={styles.sectionTitle}>Severity Distribution</Text>
      {distribution.length === 0 ? (
        <Card style={styles.empty}>
          <Ionicons name="bar-chart-outline" size={40} color={colors.textLight} />
          <Text style={styles.emptyText}>No data yet. Run a screening to populate analytics.</Text>
        </Card>
      ) : (
        <Card>
          {distribution.map(([label, count]) => {
            const pct = total > 0 ? (count / total) * 100 : 0;
            const barColor = SEVERITY_COLORS[label] ?? colors.secondary;
            return (
              <View key={label} style={styles.barRow}>
                <Text style={styles.barLabel}>{label}</Text>
                <View style={styles.barTrack}>
                  <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: barColor }]} />
                </View>
                <Text style={styles.barValue}>{count} ({Math.round(pct)}%)</Text>
              </View>
            );
          })}
        </Card>
      )}

      {/* Mode breakdown */}
      <Text style={[styles.sectionTitle, { marginTop: spacing.lg }]}>Screening Mode</Text>
      <Card>
        {Object.entries(modeCount).length === 0 ? (
          <Text style={styles.emptyText}>No data</Text>
        ) : (
          Object.entries(modeCount).map(([mode, count]) => (
            <View key={mode} style={styles.modeRow}>
              <Ionicons
                name={mode === 'image' ? 'camera-outline' : mode === 'multimodal' ? 'layers-outline' : 'calculator-outline'}
                size={18}
                color={colors.primary}
              />
              <Text style={styles.modeLabel}>{mode}</Text>
              <Text style={styles.modeCount}>{count}</Text>
            </View>
          ))
        )}
      </Card>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  pageTitle: { fontSize: 22, fontWeight: '800', color: colors.text, marginBottom: spacing.lg },
  summaryCard: { marginBottom: spacing.lg },
  summaryRow: { flexDirection: 'row', alignItems: 'center' },
  summaryItem: { flex: 1, alignItems: 'center' },
  summaryValue: { fontSize: 28, fontWeight: '800', color: colors.text },
  summaryLabel: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  summaryDivider: { width: 1, height: 40, backgroundColor: colors.border },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  empty: { alignItems: 'center', paddingVertical: spacing.xl },
  emptyText: { fontSize: 13, color: colors.textSecondary, marginTop: spacing.sm, textAlign: 'center' },
  barRow: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.sm, gap: spacing.sm },
  barLabel: { width: 80, fontSize: 12, fontWeight: '600', color: colors.text },
  barTrack: { flex: 1, height: 12, backgroundColor: colors.border, borderRadius: 6, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 6 },
  barValue: { width: 72, fontSize: 11, color: colors.textSecondary, textAlign: 'right' },
  modeRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.sm },
  modeLabel: { flex: 1, fontSize: 14, fontWeight: '500', color: colors.text, textTransform: 'capitalize' },
  modeCount: { fontSize: 14, fontWeight: '700', color: colors.primaryDark },
});
