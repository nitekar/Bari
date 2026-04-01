/**
 * edu-nutrition.tsx — Nutrients for child growth + brain development
 */
import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { NUTRIENTS } from '../src/data';

export default function NutritionScreen() {
  const router = useRouter();

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>🥦 Nutrition & Growth</Text>
      <Text style={s.subtitle}>Key nutrients every child needs for healthy development</Text>

      <View style={s.heroCard}>
        <Text style={s.heroEmoji}>🧠</Text>
        <Text style={s.heroTitle}>Why Nutrition Matters</Text>
        <Text style={s.heroText}>
          The first 1,000 days (pregnancy to age 2) shape a child's brain, immunity, and body for life. Proper nutrition during this window prevents anemia, stunting, and cognitive delays.
        </Text>
      </View>

      {NUTRIENTS.map((n) => (
        <View key={n.id} style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardEmoji}>{n.emoji}</Text>
            <Text style={s.cardTitle}>{n.nutrient}</Text>
          </View>
          <Text style={s.cardRole}>{n.role}</Text>
          <View style={s.sourcesBox}>
            <Ionicons name="restaurant-outline" size={14} color={colors.accent} />
            <Text style={s.sourcesText}>{n.sources}</Text>
          </View>
        </View>
      ))}
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
  heroCard: { backgroundColor: '#E8F5E9', borderRadius: borderRadius.lg, padding: spacing.lg, marginBottom: spacing.lg, alignItems: 'center' },
  heroEmoji: { fontSize: 40, marginBottom: 8 },
  heroTitle: { fontSize: 18, fontWeight: '700', color: colors.text, marginBottom: 8 },
  heroText: { fontSize: 14, color: colors.textSecondary, lineHeight: 21, textAlign: 'center' },
  card: { backgroundColor: colors.surface, borderRadius: borderRadius.lg, padding: spacing.md, marginBottom: spacing.sm, ...shadows.sm },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 6 },
  cardEmoji: { fontSize: 24 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: colors.text },
  cardRole: { fontSize: 13, color: colors.textSecondary, lineHeight: 19, marginBottom: 8 },
  sourcesBox: { flexDirection: 'row', alignItems: 'flex-start', gap: 6, backgroundColor: colors.accentLight, padding: spacing.sm, borderRadius: borderRadius.sm },
  sourcesText: { fontSize: 12, color: colors.text, flex: 1, lineHeight: 17 },
});
