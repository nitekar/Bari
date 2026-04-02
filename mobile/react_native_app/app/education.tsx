/**
 * app/(parent)/education.tsx — Education Hub for Parent role
 * Reuses the same hub design as (tabs)/education
 * Bundle cache invalidation...
 */
import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { useStore } from '../src/store/useStore';
import { MILESTONES, BABY_ACTIVITIES, EDUCATION_CATEGORIES } from '../src/data';
import { useTranslation } from '../src/i18n';
import { Button } from '../src/shared/components';

export default function ParentEducationHub() {
  const router = useRouter();
  const { t } = useTranslation();
  const completedItems = useStore((s) => s.completedItems);

  const allMilestones = MILESTONES.flatMap((g) => g.milestones);
  const allActivities = BABY_ACTIVITIES.flatMap((g) => g.activities);
  const totalCheckable = allMilestones.length + allActivities.length;
  const totalDone = [...allMilestones, ...allActivities].filter((item) => completedItems.includes(item.id)).length;
  const overallPct = totalCheckable > 0 ? Math.round((totalDone / totalCheckable) * 100) : 0;
  const isGuest = useStore((s) => s.userId) === null;

  if (isGuest) {
    return (
      <ScrollView style={s.scroll} contentContainerStyle={s.content}>
        {/* Context about Bari */}
        <View style={s.hero}>
          <View style={s.heroDecor1} />
          <View style={s.heroDecor2} />
          <Text style={s.heroEmoji}>💡</Text>
          <Text style={s.heroTitle}>{t.guestEducation.title}</Text>
          <Text style={s.heroSub}>{t.guestEducation.subtitle}</Text>
        </View>

        <View style={s.tipCard}>
          <View style={s.tipHeader}>
            <Ionicons name="information-circle-outline" size={20} color={colors.primaryDark} />
            <Text style={s.tipTitle}>{t.guestEducation.whatIsBari}</Text>
          </View>
          <Text style={s.tipText}>
            {t.guestEducation.whatIsBariBody}
          </Text>
        </View>

        {/* Learn Points */}
        <View style={[s.tipCard, { backgroundColor: colors.surface, marginTop: spacing.md }]}>
          <View style={s.tipHeader}>
            <Ionicons name="book-outline" size={20} color={colors.secondary} />
            <Text style={s.tipTitle}>{t.guestEducation.whyNutrition}</Text>
          </View>
          <Text style={[s.tipText, { marginBottom: 8 }]}>• {t.guestEducation.point1}</Text>
          <Text style={[s.tipText, { marginBottom: 8 }]}>• {t.guestEducation.point2}</Text>
          <Text style={[s.tipText, { marginBottom: 8 }]}>• {t.guestEducation.point3}</Text>
        </View>

        {/* Call to Action */}
        <View style={{ marginTop: spacing.xl, paddingHorizontal: spacing.md }}>
          <Text style={{ textAlign: 'center', color: colors.textSecondary, marginBottom: spacing.md, lineHeight: 22 }}>
            {t.guestEducation.ctaBody}
          </Text>
          <Button
            title={t.guestEducation.getStarted}
            onPress={() => router.push('/auth')}
            variant="primary"
            icon={<Ionicons name="person-add-outline" size={20} color={colors.white} />}
          />
        </View>
        <View style={{ height: 100 }} />
      </ScrollView>
    );
  }

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content}>
      {/* Hero */}
      <View style={s.hero}>
        <View style={s.heroDecor1} />
        <View style={s.heroDecor2} />
        <Text style={s.heroEmoji}>📚</Text>
        <Text style={s.heroTitle}>{t.education.heroTitle}</Text>
        <Text style={s.heroSub}>{t.education.heroSub}</Text>
        {totalDone > 0 && (
          <View style={s.progressBox}>
            <View style={s.progressBarBg}>
              <View style={[s.progressBarFill, { width: `${overallPct}%` }]} />
            </View>
            <Text style={s.progressLabel}>{totalDone}/{totalCheckable} {t.education.progressCompleted}</Text>
          </View>
        )}
      </View>

      {/* Category grid */}
      <View style={s.grid}>
        {EDUCATION_CATEGORIES.map((cat) => {
          // Map ID directly to translation block dynamically
          const localizedId = cat.id.replace('cat-', '') as keyof typeof t.education.categories;
          const translatedCat = t.education.categories[localizedId];

          return (
            <TouchableOpacity
              key={cat.id}
              style={[s.catCard, { backgroundColor: cat.bg }]}
              onPress={() => router.push(cat.route as any)}
              activeOpacity={0.75}
            >
              <View style={[s.catIconBox, { backgroundColor: cat.color + '25' }]}>
                <Text style={s.catEmoji}>{cat.emoji}</Text>
              </View>
              <Text style={[s.catTitle, { color: cat.color }]}>{translatedCat?.title || cat.title}</Text>
              <Text style={s.catSub}>{translatedCat?.subtitle || cat.subtitle}</Text>
              <View style={s.catArrow}>
                <Ionicons name="arrow-forward" size={14} color={cat.color} />
              </View>
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={s.tipCard}>
        <View style={s.tipHeader}>
          <Ionicons name="bulb-outline" size={20} color={colors.warning} />
          <Text style={s.tipTitle}>{t.education.tipTitle}</Text>
        </View>
        <Text style={s.tipText}>{t.education.tipContent}</Text>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const s = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingTop: 16 },
  hero: { backgroundColor: '#E8849A', borderRadius: borderRadius.xl, padding: spacing.xl, marginBottom: spacing.lg, overflow: 'hidden', alignItems: 'center', position: 'relative' },
  heroDecor1: { position: 'absolute', top: -30, right: -20, width: 100, height: 100, borderRadius: 50, backgroundColor: 'rgba(255,255,255,0.1)' },
  heroDecor2: { position: 'absolute', bottom: -20, left: -15, width: 80, height: 80, borderRadius: 40, backgroundColor: 'rgba(255,255,255,0.08)' },
  heroEmoji: { fontSize: 40, marginBottom: 8 },
  heroTitle: { fontSize: 24, fontWeight: '800', color: colors.white, marginBottom: 6 },
  heroSub: { fontSize: 14, color: 'rgba(255,255,255,0.85)', textAlign: 'center', lineHeight: 20 },
  progressBox: { marginTop: spacing.md, width: '100%' },
  progressBarBg: { height: 6, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 3, overflow: 'hidden' },
  progressBarFill: { height: 6, backgroundColor: colors.white, borderRadius: 3 },
  progressLabel: { fontSize: 11, color: 'rgba(255,255,255,0.7)', textAlign: 'center', marginTop: 4 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  catCard: { width: '48%', borderRadius: borderRadius.lg, padding: spacing.md, ...shadows.sm, position: 'relative', minHeight: 130 },
  catIconBox: { width: 44, height: 44, borderRadius: 14, alignItems: 'center', justifyContent: 'center', marginBottom: spacing.sm },
  catEmoji: { fontSize: 22 },
  catTitle: { fontSize: 14, fontWeight: '700', marginBottom: 2 },
  catSub: { fontSize: 11, color: colors.textSecondary, lineHeight: 15 },
  catArrow: { position: 'absolute', bottom: spacing.sm, right: spacing.sm },
  tipCard: { backgroundColor: '#FFF3E0', borderRadius: borderRadius.lg, padding: spacing.md, marginTop: spacing.lg },
  tipHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 },
  tipTitle: { fontSize: 14, fontWeight: '700', color: colors.text },
  tipText: { fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
});
