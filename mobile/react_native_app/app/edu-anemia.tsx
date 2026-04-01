/**
 * edu-anemia.tsx — Anemia education: causes, symptoms, prevention
 */
import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { ANEMIA_SECTIONS } from '../src/data';

export default function AnemiaScreen() {
  const router = useRouter();

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      <TouchableOpacity onPress={() => router.back()} style={s.backBtn}>
        <Ionicons name="arrow-back" size={22} color={colors.primaryDark} />
        <Text style={s.backText}>Back</Text>
      </TouchableOpacity>

      <Text style={s.title}>🩸 Anemia Guide</Text>
      <Text style={s.subtitle}>Understanding anemia in children</Text>

      <View style={s.alertCard}>
        <Ionicons name="information-circle" size={22} color={colors.error} />
        <Text style={s.alertText}>
          Anemia is the #1 nutritional disorder in children under 5 in Africa. Early detection and iron-rich diets can prevent serious health consequences.
        </Text>
      </View>

      {ANEMIA_SECTIONS.map((section) => (
        <View key={section.id} style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardEmoji}>{section.emoji}</Text>
            <Text style={s.cardTitle}>{section.title}</Text>
          </View>
          {section.points.map((point, i) => (
            <View key={i} style={s.pointRow}>
              <View style={s.bullet} />
              <Text style={s.pointText}>{point}</Text>
            </View>
          ))}
        </View>
      ))}

      <View style={s.ctaCard}>
        <Text style={s.ctaEmoji}>📱</Text>
        <Text style={s.ctaTitle}>Screen your child today</Text>
        <Text style={s.ctaText}>Use Bari's AI-powered conjunctiva scan to check for anemia in under 30 seconds.</Text>
        <TouchableOpacity style={s.ctaBtn} onPress={() => router.replace('/')}>
          <Ionicons name="medkit-outline" size={18} color={colors.white} />
          <Text style={s.ctaBtnText}>Start Screening</Text>
        </TouchableOpacity>
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
  alertCard: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, backgroundColor: '#FFEBEE', borderRadius: borderRadius.lg, padding: spacing.md, marginBottom: spacing.lg, borderLeftWidth: 4, borderLeftColor: colors.error },
  alertText: { fontSize: 13, color: colors.text, flex: 1, lineHeight: 20 },
  card: { backgroundColor: colors.surface, borderRadius: borderRadius.lg, padding: spacing.md, marginBottom: spacing.md, ...shadows.sm },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: spacing.sm },
  cardEmoji: { fontSize: 24 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: colors.text },
  pointRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, paddingVertical: 4 },
  bullet: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.primary, marginTop: 6 },
  pointText: { fontSize: 13, color: colors.textSecondary, flex: 1, lineHeight: 19 },
  ctaCard: { backgroundColor: colors.primary, borderRadius: borderRadius.xl, padding: spacing.xl, alignItems: 'center', marginTop: spacing.md },
  ctaEmoji: { fontSize: 36, marginBottom: 8 },
  ctaTitle: { fontSize: 18, fontWeight: '700', color: colors.white, marginBottom: 6 },
  ctaText: { fontSize: 13, color: 'rgba(255,255,255,0.85)', textAlign: 'center', lineHeight: 19, marginBottom: spacing.md },
  ctaBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: spacing.lg, paddingVertical: 12, borderRadius: borderRadius.pill },
  ctaBtnText: { fontSize: 15, fontWeight: '700', color: colors.white },
});
