/**
 * edu-feedingplan.tsx — Localized weekly feeding plan (Rwanda/Africa)
 */
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { WEEKLY_FEEDING_PLAN } from '../src/data';

export default function FeedingPlanScreen() {
  const router = useRouter();
  const today = new Date().getDay(); // 0=Sun, 1=Mon...
  const todayIdx = today === 0 ? 6 : today - 1; // map to 0=Mon
  const [selectedDay, setSelectedDay] = useState(todayIdx);

  const dayPlan = WEEKLY_FEEDING_PLAN[selectedDay];

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>🍽️ Weekly Meal Plan</Text>
      <Text style={s.subtitle}>Balanced Rwandan meals for healthy children</Text>

      {/* Day selector */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.dayScroll}>
        {WEEKLY_FEEDING_PLAN.map((day, i) => (
          <TouchableOpacity
            key={day.id}
            style={[s.dayChip, selectedDay === i && s.dayChipActive]}
            onPress={() => setSelectedDay(i)}
          >
            <Text style={[s.dayText, selectedDay === i && s.dayTextActive]}>
              {day.day.slice(0, 3)}
            </Text>
            {i === todayIdx && <View style={s.todayDot} />}
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Day header */}
      <View style={s.dayHeader}>
        <Text style={s.dayTitle}>{dayPlan.day}</Text>
        {selectedDay === todayIdx && (
          <View style={s.todayBadge}><Text style={s.todayText}>Today</Text></View>
        )}
      </View>

      {/* Meals */}
      {dayPlan.meals.map((meal, i) => (
        <View key={i} style={s.mealCard}>
          <View style={s.mealTime}>
            <Text style={s.mealEmoji}>{meal.emoji}</Text>
            <Text style={s.mealTimeText}>{meal.time}</Text>
          </View>
          <View style={s.mealBody}>
            <Text style={s.mealName}>{meal.name}</Text>
            <Text style={s.mealDesc}>{meal.description}</Text>
          </View>
        </View>
      ))}

      {/* Tips */}
      <View style={s.tipCard}>
        <Ionicons name="bulb-outline" size={18} color={colors.warning} />
        <View style={{ flex: 1 }}>
          <Text style={s.tipTitle}>Nutrition Tips</Text>
          <Text style={s.tipText}>• Add lemon or orange to meals — vitamin C boosts iron absorption 3x</Text>
          <Text style={s.tipText}>• Liver is the richest iron source — include 1-2x per week</Text>
          <Text style={s.tipText}>• Breastmilk remains important until 2 years old</Text>
        </View>
      </View>
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
  dayScroll: { marginBottom: spacing.md },
  dayChip: { alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 20, marginRight: 8, backgroundColor: colors.surface, ...shadows.sm },
  dayChipActive: { backgroundColor: colors.primary },
  dayText: { fontSize: 13, fontWeight: '700', color: colors.textSecondary },
  dayTextActive: { color: colors.white },
  todayDot: { width: 5, height: 5, borderRadius: 3, backgroundColor: colors.primary, marginTop: 3 },
  dayHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: spacing.md },
  dayTitle: { fontSize: 20, fontWeight: '800', color: colors.text },
  todayBadge: { backgroundColor: colors.successBg, paddingHorizontal: 10, paddingVertical: 3, borderRadius: 10 },
  todayText: { fontSize: 11, fontWeight: '700', color: colors.success },
  mealCard: { flexDirection: 'row', backgroundColor: colors.surface, borderRadius: borderRadius.lg, padding: spacing.md, marginBottom: spacing.sm, ...shadows.sm },
  mealTime: { alignItems: 'center', width: 60, marginRight: spacing.sm },
  mealEmoji: { fontSize: 24, marginBottom: 4 },
  mealTimeText: { fontSize: 10, fontWeight: '600', color: colors.textLight, textAlign: 'center' },
  mealBody: { flex: 1 },
  mealName: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: 3 },
  mealDesc: { fontSize: 13, color: colors.textSecondary, lineHeight: 18 },
  tipCard: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, backgroundColor: '#FFF3E0', borderRadius: borderRadius.lg, padding: spacing.md, marginTop: spacing.md },
  tipTitle: { fontSize: 14, fontWeight: '700', color: colors.text, marginBottom: 4 },
  tipText: { fontSize: 12, color: colors.textSecondary, lineHeight: 18, marginBottom: 2 },
});
