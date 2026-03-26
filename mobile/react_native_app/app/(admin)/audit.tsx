/**
 * app/(admin)/audit.tsx — Recent prediction logs with filter tabs
 */
import React, { useState, useMemo } from 'react';
import { ScrollView, View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';

type FilterTab = 'All' | 'Severe' | 'Moderate';
const TABS: FilterTab[] = ['All', 'Severe', 'Moderate'];

function severityColor(p: string) {
  if (p === 'Severe') return colors.error;
  if (p === 'Moderate') return '#e67e22';
  if (p === 'Mild') return '#f1c40f';
  return colors.primary;
}

export default function AuditScreen() {
  const history = useStore((s) => s.history);
  const [activeTab, setActiveTab] = useState<FilterTab>('All');

  const filtered = useMemo(() => {
    if (activeTab === 'All') return history;
    return history.filter(r => r.prediction === activeTab);
  }, [history, activeTab]);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.pageTitle}>Audit Log</Text>

      {/* Filter tabs */}
      <View style={styles.tabRow}>
        {TABS.map(tab => (
          <TouchableOpacity
            key={tab}
            style={[styles.tab, activeTab === tab && styles.tabActive]}
            onPress={() => setActiveTab(tab)}
            activeOpacity={0.7}
          >
            <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>{tab}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {filtered.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="document-text-outline" size={48} color={colors.textLight} />
          <Text style={styles.emptyText}>No records for this filter</Text>
        </View>
      ) : (
        filtered.map(r => (
          <Card key={r.id} style={styles.card}>
            <View style={styles.headerRow}>
              <View style={[styles.badge, { backgroundColor: severityColor(r.prediction) + '20' }]}>
                <Text style={[styles.badgeText, { color: severityColor(r.prediction) }]}>{r.prediction}</Text>
              </View>
              <Text style={styles.timestamp}>{new Date(r.date).toLocaleString()}</Text>
            </View>
            <View style={styles.detailRow}>
              <Ionicons name="person-outline" size={14} color={colors.textLight} />
              <Text style={styles.detail}>{r.patientName ?? 'Anonymous'}</Text>
            </View>
            <View style={styles.detailRow}>
              <Ionicons name="layers-outline" size={14} color={colors.textLight} />
              <Text style={styles.detail}>{r.mode} · {Math.round(r.confidence * 100)}% confidence</Text>
            </View>
            {r.patientLocation && (
              <View style={styles.detailRow}>
                <Ionicons name="location-outline" size={14} color={colors.textLight} />
                <Text style={styles.detail}>{r.patientLocation}</Text>
              </View>
            )}
            <View style={styles.idRow}>
              <Text style={styles.idText}>ID: {r.id}</Text>
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
  pageTitle: { fontSize: 22, fontWeight: '800', color: colors.text, marginBottom: spacing.md },
  tabRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.lg },
  tab: {
    paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    borderRadius: borderRadius.pill, borderWidth: 1,
    borderColor: colors.border, backgroundColor: colors.surface,
  },
  tabActive: { backgroundColor: colors.primaryDark, borderColor: colors.primaryDark },
  tabText: { fontSize: 13, fontWeight: '600', color: colors.textSecondary },
  tabTextActive: { color: colors.white },
  empty: { alignItems: 'center', paddingVertical: spacing.xxl },
  emptyText: { fontSize: 15, color: colors.textSecondary, marginTop: spacing.md },
  card: { marginBottom: spacing.sm },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.sm },
  badge: { paddingHorizontal: spacing.sm, paddingVertical: 3, borderRadius: borderRadius.pill },
  badgeText: { fontSize: 11, fontWeight: '700' },
  timestamp: { fontSize: 11, color: colors.textLight },
  detailRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: 3 },
  detail: { fontSize: 13, color: colors.textSecondary },
  idRow: { marginTop: spacing.xs },
  idText: { fontSize: 10, color: colors.textLight, fontFamily: 'monospace' },
});
