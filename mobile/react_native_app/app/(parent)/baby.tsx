/**
 * app/(parent)/baby.tsx — Baby Monitor Hub
 * 3 cards: Sleep Tracking, Feeding Log, Development Stage.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';

export default function BabyScreen() {
  const router = useRouter();
  const sleepLogs = useStore((s) => s.sleepLogs);
  const feedingLogs = useStore((s) => s.feedingLogs);
  const childAgeMonths = useStore((s) => s.childAgeMonths);

  const lastSleep = sleepLogs[0];
  const lastFeeding = feedingLogs[0];

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.pageTitle}>Baby Monitor</Text>
      <Text style={styles.pageSub}>Track sleep, feeding, and development milestones.</Text>

      {/* Sleep Tracking */}
      <HubCard
        icon="moon"
        iconColor={colors.primaryDark}
        title="Sleep Tracking"
        description="Log sleep sessions and see patterns."
        lastEntry={lastSleep
          ? `${lastSleep.durationHours.toFixed(1)}h · ${new Date(lastSleep.date).toLocaleDateString()}`
          : 'No sleep entries yet'}
        count={sleepLogs.length}
        countLabel="sessions"
        onAdd={() => router.push('/parent-sleep')}
        onView={() => router.push('/parent-sleep')}
      />

      {/* Feeding Log */}
      <HubCard
        icon="nutrition"
        iconColor={colors.secondary}
        title="Feeding Log"
        description="Track breastfeeding, formula, and solid foods."
        lastEntry={lastFeeding
          ? `${lastFeeding.type} at ${lastFeeding.time}`
          : 'No feeding entries yet'}
        count={feedingLogs.length}
        countLabel="entries"
        onAdd={() => router.push('/parent-feeding')}
        onView={() => router.push('/parent-feeding')}
      />

      {/* Development Stage */}
      <HubCard
        icon="trending-up"
        iconColor={colors.accentDark}
        title="Development Stage"
        description="Age-appropriate milestones and feeding guidance."
        lastEntry={childAgeMonths != null
          ? `${childAgeMonths} months old`
          : 'Set child age in Profile'}
        count={null}
        countLabel=""
        onAdd={() => router.push('/parent-development')}
        onView={() => router.push('/parent-development')}
        addLabel="View Milestones"
      />

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function HubCard({
  icon, iconColor, title, description, lastEntry,
  count, countLabel, onAdd, onView, addLabel,
}: {
  icon: any; iconColor: string; title: string; description: string;
  lastEntry: string; count: number | null; countLabel: string;
  onAdd: () => void; onView: () => void; addLabel?: string;
}) {
  return (
    <Card style={styles.hubCard}>
      <View style={styles.hubHeader}>
        <View style={[styles.hubIcon, { backgroundColor: iconColor + '20' }]}>
          <Ionicons name={icon} size={24} color={iconColor} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.hubTitle}>{title}</Text>
          <Text style={styles.hubDesc}>{description}</Text>
        </View>
        {count !== null && (
          <View style={styles.countBadge}>
            <Text style={styles.countValue}>{count}</Text>
            <Text style={styles.countLabel}>{countLabel}</Text>
          </View>
        )}
      </View>
      <View style={styles.hubEntry}>
        <Ionicons name="time-outline" size={14} color={colors.textLight} />
        <Text style={styles.hubEntryText}>{lastEntry}</Text>
      </View>
      <View style={styles.hubActions}>
        <TouchableOpacity style={[styles.actionBtn, styles.actionBtnPrimary, { backgroundColor: iconColor }]} onPress={onAdd} activeOpacity={0.8}>
          <Ionicons name="add-outline" size={16} color={colors.white} />
          <Text style={styles.actionBtnTextPrimary}>{addLabel ?? 'Add Entry'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionBtn, styles.actionBtnSecondary]} onPress={onView} activeOpacity={0.8}>
          <Text style={styles.actionBtnText}>View All</Text>
        </TouchableOpacity>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  pageTitle: { fontSize: 22, fontWeight: '800', color: colors.text, marginBottom: 4 },
  pageSub: { fontSize: 14, color: colors.textSecondary, marginBottom: spacing.lg },
  hubCard: { marginBottom: spacing.md },
  hubHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.md, marginBottom: spacing.md },
  hubIcon: { width: 48, height: 48, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  hubTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: 2 },
  hubDesc: { fontSize: 12, color: colors.textSecondary },
  countBadge: { alignItems: 'center', backgroundColor: colors.surfaceElevated, paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: borderRadius.md },
  countValue: { fontSize: 18, fontWeight: '800', color: colors.text },
  countLabel: { fontSize: 10, color: colors.textSecondary },
  hubEntry: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: spacing.md, paddingVertical: spacing.xs, borderTopWidth: 1, borderTopColor: colors.border },
  hubEntryText: { fontSize: 12, color: colors.textSecondary, flex: 1 },
  hubActions: { flexDirection: 'row', gap: spacing.sm },
  actionBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 10, borderRadius: borderRadius.md, gap: 4 },
  actionBtnPrimary: {},
  actionBtnSecondary: { backgroundColor: colors.surfaceElevated },
  actionBtnTextPrimary: { fontSize: 13, fontWeight: '700', color: colors.white },
  actionBtnText: { fontSize: 13, fontWeight: '600', color: colors.text },
});
