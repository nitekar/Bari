/**
 * edu-activities.tsx — Brain-boosting baby activities by age
 */
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { useStore } from '../src/store/useStore';
import { BABY_ACTIVITIES } from '../src/data';

export default function ActivitiesScreen() {
  const router = useRouter();
  const completedItems = useStore((s) => s.completedItems);
  const toggleCompleted = useStore((s) => s.toggleCompleted);
  const [expandedId, setExpandedId] = useState<string | null>(BABY_ACTIVITIES[0].id);

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>🧸 Baby Activities</Text>
      <Text style={s.subtitle}>Fun activities to boost brain development</Text>

      {BABY_ACTIVITIES.map((group) => {
        const done = group.activities.filter((a) => completedItems.includes(a.id)).length;
        const total = group.activities.length;
        const isOpen = expandedId === group.id;

        return (
          <View key={group.id} style={s.card}>
            <TouchableOpacity
              style={s.cardHeader}
              onPress={() => setExpandedId(isOpen ? null : group.id)}
              activeOpacity={0.7}
            >
              <Text style={s.emoji}>{group.emoji}</Text>
              <View style={{ flex: 1 }}>
                <Text style={s.groupAge}>{group.ageRange}</Text>
                <Text style={s.groupSub}>{group.subtitle}</Text>
              </View>
              <View style={s.badge}>
                <Text style={s.badgeText}>{done}/{total}</Text>
              </View>
              <Ionicons name={isOpen ? 'chevron-up' : 'chevron-down'} size={20} color={colors.textLight} />
            </TouchableOpacity>

            {isOpen && group.activities.map((act) => {
              const checked = completedItems.includes(act.id);
              return (
                <TouchableOpacity
                  key={act.id}
                  style={s.actRow}
                  onPress={() => toggleCompleted(act.id)}
                  activeOpacity={0.6}
                >
                  <View style={[s.checkbox, checked && s.checkboxDone]}>
                    {checked && <Ionicons name="checkmark" size={14} color={colors.white} />}
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={s.actTitle}>{act.emoji} {act.title}</Text>
                    <Text style={s.actDesc}>{act.description}</Text>
                  </View>
                </TouchableOpacity>
              );
            })}
          </View>
        );
      })}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const s = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingTop: 52 },
  backBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: spacing.md },
  backText: { fontSize: 15, fontWeight: '600', color: colors.primaryDark },
  title: { fontSize: 26, fontWeight: '800', color: colors.text, marginBottom: 4 },
  subtitle: { fontSize: 14, color: colors.textSecondary, marginBottom: spacing.lg },
  card: { backgroundColor: colors.surface, borderRadius: borderRadius.lg, marginBottom: spacing.md, ...shadows.md, overflow: 'hidden' },
  cardHeader: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, gap: spacing.sm },
  emoji: { fontSize: 28 },
  groupAge: { fontSize: 15, fontWeight: '700', color: colors.text },
  groupSub: { fontSize: 12, color: colors.textSecondary },
  badge: { backgroundColor: colors.primaryLight, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10 },
  badgeText: { fontSize: 11, fontWeight: '700', color: colors.primaryDark },
  actRow: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm, paddingHorizontal: spacing.md, paddingVertical: 10, borderTopWidth: 1, borderTopColor: colors.borderLight },
  checkbox: { width: 22, height: 22, borderRadius: 6, borderWidth: 2, borderColor: colors.border, alignItems: 'center', justifyContent: 'center', marginTop: 2 },
  checkboxDone: { backgroundColor: colors.success, borderColor: colors.success },
  actTitle: { fontSize: 14, fontWeight: '600', color: colors.text, marginBottom: 2 },
  actDesc: { fontSize: 12, color: colors.textSecondary, lineHeight: 17 },
});
