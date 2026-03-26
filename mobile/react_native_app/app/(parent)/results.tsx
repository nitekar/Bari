/**
 * app/(parent)/results.tsx — Screening history for parents
 * Simplified: risk badge, plain-language action guidance, no confidence %.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';

interface RiskInfo {
  label: string;
  color: string;
  bg: string;
  icon: keyof typeof Ionicons.glyphMap;
  guidance: string;
}

function getRisk(prediction: string): RiskInfo {
  switch (prediction) {
    case 'Severe':
    case 'Anemic':
      return {
        label: 'High Risk',
        color: colors.error,
        bg: colors.error + '18',
        icon: 'alert-circle',
        guidance: 'Please visit a health facility immediately. Your child may need medical treatment.',
      };
    case 'Moderate':
      return {
        label: 'Moderate Risk',
        color: '#e67e22',
        bg: '#e67e2218',
        icon: 'warning',
        guidance: 'Schedule an appointment with your community health worker within the next few days.',
      };
    case 'Mild':
      return {
        label: 'Low Risk',
        color: '#c8a000',
        bg: '#f1c40f18',
        icon: 'information-circle',
        guidance: 'Increase iron-rich foods in your child\'s diet. Re-screen in 30 days.',
      };
    default:
      return {
        label: 'Healthy',
        color: colors.primary,
        bg: colors.primary + '18',
        icon: 'checkmark-circle',
        guidance: 'Great! Continue with a balanced, iron-rich diet and regular check-ups.',
      };
  }
}

export default function ParentResultsScreen() {
  const history = useStore((s) => s.history);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.pageTitle}>Screening Results</Text>
      <Text style={styles.pageSub}>Your child's anemia screening history.</Text>

      {history.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="pulse-outline" size={48} color={colors.textLight} />
          <Text style={styles.emptyTitle}>No Results Yet</Text>
          <Text style={styles.emptySub}>Complete a screening to see results here.</Text>
        </View>
      ) : (
        history.map(r => {
          const risk = getRisk(r.prediction);
          return (
            <Card key={r.id} style={[styles.card, { borderLeftColor: risk.color, borderLeftWidth: 4 }]}>
              <View style={styles.cardHeader}>
                <View style={[styles.badge, { backgroundColor: risk.bg }]}>
                  <Ionicons name={risk.icon} size={14} color={risk.color} />
                  <Text style={[styles.badgeText, { color: risk.color }]}>{risk.label}</Text>
                </View>
                <Text style={styles.date}>{new Date(r.date).toLocaleDateString()}</Text>
              </View>
              <Text style={styles.guidance}>{risk.guidance}</Text>
              {r.patientName && (
                <View style={styles.nameRow}>
                  <Ionicons name="person-outline" size={12} color={colors.textLight} />
                  <Text style={styles.nameText}>{r.patientName}</Text>
                </View>
              )}
            </Card>
          );
        })
      )}

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  pageTitle: { fontSize: 22, fontWeight: '800', color: colors.text, marginBottom: 4 },
  pageSub: { fontSize: 14, color: colors.textSecondary, marginBottom: spacing.lg },
  empty: { alignItems: 'center', paddingVertical: spacing.xxl },
  emptyTitle: { fontSize: 16, fontWeight: '600', color: colors.textSecondary, marginTop: spacing.md },
  emptySub: { fontSize: 13, color: colors.textLight, marginTop: 4 },
  card: { marginBottom: spacing.md },
  cardHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.sm },
  badge: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: borderRadius.pill },
  badgeText: { fontSize: 12, fontWeight: '700' },
  date: { fontSize: 12, color: colors.textLight },
  guidance: { fontSize: 14, color: colors.textSecondary, lineHeight: 21 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: spacing.sm },
  nameText: { fontSize: 12, color: colors.textLight },
});
