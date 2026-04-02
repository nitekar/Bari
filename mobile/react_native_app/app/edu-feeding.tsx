/**
 * edu-feeding.tsx — Feeding stages guide by age group
 */
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { FEEDING_STAGES } from '../src/data';

export default function FeedingGuideScreen() {
  const router = useRouter();
  const [expandedId, setExpandedId] = useState<string | null>(FEEDING_STAGES[0].id);

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>🥣 Feeding Guide</Text>
      <Text style={s.subtitle}>What to feed your child at each stage</Text>

      {FEEDING_STAGES.map((stage) => {
        const isOpen = expandedId === stage.id;
        return (
          <View key={stage.id} style={s.card}>
            <TouchableOpacity
              style={s.cardHeader}
              onPress={() => setExpandedId(isOpen ? null : stage.id)}
              activeOpacity={0.7}
            >
              <Text style={s.emoji}>{stage.emoji}</Text>
              <View style={{ flex: 1 }}>
                <Text style={s.stageAge}>{stage.ageRange}</Text>
                <Text style={s.stageTexture}>{stage.texture}</Text>
              </View>
              <Ionicons name={isOpen ? 'chevron-up' : 'chevron-down'} size={20} color={colors.textLight} />
            </TouchableOpacity>

            {isOpen && (
              <View style={s.cardBody}>
                <View style={s.infoRow}>
                  <Ionicons name="time-outline" size={16} color={colors.primary} />
                  <Text style={s.infoText}>{stage.frequency}</Text>
                </View>
                <View style={s.tipBox}>
                  <Ionicons name="bulb-outline" size={16} color={colors.warning} />
                  <Text style={s.tipText}>{stage.tips}</Text>
                </View>
                <Text style={s.foodsLabel}>Foods to introduce:</Text>
                {stage.foods.map((f) => (
                  <View key={f.id} style={s.foodRow}>
                    <Text style={s.foodEmoji}>{f.emoji}</Text>
                    <Text style={s.foodName}>{f.food}</Text>
                  </View>
                ))}
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
  card: { backgroundColor: colors.surface, borderRadius: borderRadius.lg, marginBottom: spacing.md, ...shadows.md, overflow: 'hidden' },
  cardHeader: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, gap: spacing.md },
  emoji: { fontSize: 32 },
  stageAge: { fontSize: 16, fontWeight: '700', color: colors.text },
  stageTexture: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  cardBody: { paddingHorizontal: spacing.md, paddingBottom: spacing.md },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  infoText: { fontSize: 13, color: colors.text, fontWeight: '600' },
  tipBox: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, backgroundColor: colors.warningBg, padding: spacing.sm, borderRadius: borderRadius.sm, marginBottom: spacing.md },
  tipText: { fontSize: 13, color: colors.text, flex: 1, lineHeight: 19 },
  foodsLabel: { fontSize: 13, fontWeight: '700', color: colors.textSecondary, marginBottom: 6 },
  foodRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: colors.borderLight },
  foodEmoji: { fontSize: 20 },
  foodName: { fontSize: 14, color: colors.text },
});
