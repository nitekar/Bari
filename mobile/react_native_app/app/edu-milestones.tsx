/**
 * edu-milestones.tsx — Child development milestones with interactive checklists
 */
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { useStore } from '../src/store/useStore';
import { MILESTONES } from '../src/data';

const CAT_COLORS = { motor: '#FF8A65', cognitive: '#42A5F5', social: '#AB47BC' };
const CAT_LABELS = { motor: '💪 Motor', cognitive: '🧠 Cognitive', social: '❤️ Social' };

export default function MilestonesScreen() {
  const router = useRouter();
  const completedItems = useStore((s) => s.completedItems);
  const toggleCompleted = useStore((s) => s.toggleCompleted);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(MILESTONES[0].id);

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>📋 Child Milestones</Text>
      <Text style={s.subtitle}>Track your child's development by age group</Text>

      {MILESTONES.map((group) => {
        const total = group.milestones.length;
        const done = group.milestones.filter((m) => completedItems.includes(m.id)).length;
        const pct = total > 0 ? Math.round((done / total) * 100) : 0;
        const isOpen = expandedGroup === group.id;

        return (
          <View key={group.id} style={s.groupCard}>
            <TouchableOpacity
              style={s.groupHeader}
              onPress={() => setExpandedGroup(isOpen ? null : group.id)}
              activeOpacity={0.7}
            >
              <Text style={s.groupEmoji}>{group.emoji}</Text>
              <View style={{ flex: 1 }}>
                <Text style={s.groupTitle}>{group.ageRange}</Text>
                <View style={s.progressRow}>
                  <View style={s.progressBg}>
                    <View style={[s.progressFill, { width: `${pct}%`, backgroundColor: pct === 100 ? colors.success : colors.primary }]} />
                  </View>
                  <Text style={s.progressText}>{done}/{total}</Text>
                </View>
              </View>
              <Ionicons name={isOpen ? 'chevron-up' : 'chevron-down'} size={20} color={colors.textLight} />
            </TouchableOpacity>

            {isOpen && (
              <View style={s.itemList}>
                {(['motor', 'cognitive', 'social'] as const).map((cat) => {
                  const items = group.milestones.filter((m) => m.category === cat);
                  if (items.length === 0) return null;
                  return (
                    <View key={cat}>
                      <Text style={[s.catLabel, { color: CAT_COLORS[cat] }]}>{CAT_LABELS[cat]}</Text>
                      {items.map((m) => {
                        const checked = completedItems.includes(m.id);
                        return (
                          <TouchableOpacity
                            key={m.id}
                            style={s.checkRow}
                            onPress={() => toggleCompleted(m.id)}
                            activeOpacity={0.6}
                          >
                            <View style={[s.checkbox, checked && { backgroundColor: colors.success, borderColor: colors.success }]}>
                              {checked && <Ionicons name="checkmark" size={14} color={colors.white} />}
                            </View>
                            <Text style={[s.checkLabel, checked && s.checkLabelDone]}>{m.label}</Text>
                          </TouchableOpacity>
                        );
                      })}
                    </View>
                  );
                })}
              </View>
            )}
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
  groupCard: { backgroundColor: colors.surface, borderRadius: borderRadius.lg, marginBottom: spacing.md, ...shadows.md, overflow: 'hidden' },
  groupHeader: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, gap: spacing.md },
  groupEmoji: { fontSize: 32 },
  groupTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: 6 },
  progressRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  progressBg: { flex: 1, height: 6, backgroundColor: colors.borderLight, borderRadius: 3, overflow: 'hidden' },
  progressFill: { height: 6, borderRadius: 3 },
  progressText: { fontSize: 12, fontWeight: '700', color: colors.textSecondary, width: 36 },
  itemList: { paddingHorizontal: spacing.md, paddingBottom: spacing.md },
  catLabel: { fontSize: 12, fontWeight: '700', marginTop: spacing.sm, marginBottom: 6 },
  checkRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8, gap: spacing.sm },
  checkbox: { width: 22, height: 22, borderRadius: 6, borderWidth: 2, borderColor: colors.border, alignItems: 'center', justifyContent: 'center' },
  checkLabel: { fontSize: 14, color: colors.text, flex: 1 },
  checkLabelDone: { color: colors.textLight, textDecorationLine: 'line-through' },
});
