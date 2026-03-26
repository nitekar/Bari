/**
 * app/parent-development.tsx — Development milestones by age
 * Shows current stage + physical, cognitive, feeding readiness milestones.
 */
import React from 'react';
import { ScrollView, View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../src/shared/theme';
import { Card, Button } from '../src/shared/components';
import { useStore } from '../src/store/useStore';

interface MilestoneStage {
  label: string;
  minMonths: number;
  maxMonths: number;
  color: string;
  physical: string[];
  cognitive: string[];
  feeding: string[];
}

const STAGES: MilestoneStage[] = [
  {
    label: '6–8 Months',
    minMonths: 6, maxMonths: 8,
    color: '#4CAF50',
    physical: ['Rolls over both ways', 'Sits with support', 'Bears weight on legs when held standing'],
    cognitive: ['Recognises familiar faces', 'Babbles and imitates sounds', 'Explores objects with mouth'],
    feeding: ['Ready to start solids (pureed foods)', 'Iron-fortified cereals and pureed vegetables', 'Continue breastfeeding/formula as main nutrition'],
  },
  {
    label: '9–12 Months',
    minMonths: 9, maxMonths: 11,
    color: '#2196F3',
    physical: ['Pulls to stand', 'Crawls or bottom shuffles', 'Pincer grasp developing'],
    cognitive: ['Understands "no"', 'Waves bye-bye', 'Searches for hidden objects'],
    feeding: ['Progresses to mashed and lumpy foods', 'Introduce finger foods', 'Offer liver, eggs, beans for iron'],
  },
  {
    label: '12–18 Months',
    minMonths: 12, maxMonths: 17,
    color: '#9C27B0',
    physical: ['Walks independently', 'Climbs stairs with support', 'Stacks 2–3 blocks'],
    cognitive: ['Says 1–3 words', 'Points to desired objects', 'Imitates simple actions'],
    feeding: ['Eats family foods (soft and chopped)', 'Self-feeds with fingers', 'Needs iron-rich foods daily'],
  },
  {
    label: '18–24 Months',
    minMonths: 18, maxMonths: 23,
    color: '#FF9800',
    physical: ['Runs steadily', 'Kicks a ball', 'Scribbles with crayons'],
    cognitive: ['Vocabulary 20–50 words', 'Two-word phrases emerging', 'Parallel play with other children'],
    feeding: ['3 main meals + 2 snacks daily', 'Include variety across all food groups', 'Small portions, frequent offering'],
  },
  {
    label: '24–36 Months',
    minMonths: 24, maxMonths: 35,
    color: '#E91E63',
    physical: ['Jumps with both feet', 'Pedals a tricycle', 'Uses spoon and fork'],
    cognitive: ['2–3 word sentences', 'Names familiar objects', 'Understands simple stories'],
    feeding: ['Eats most family foods', 'May be picky — keep offering varied foods', 'Animal proteins 3–4 times per week for iron'],
  },
  {
    label: '36–48 Months',
    minMonths: 36, maxMonths: 47,
    color: '#00BCD4',
    physical: ['Hops on one foot', 'Draws simple shapes', 'Dresses with minimal help'],
    cognitive: ['4–5 word sentences', 'Asks "why" questions', 'Understands turn-taking'],
    feeding: ['Appetite may decrease (growth slows)', 'Offer iron-rich foods at every meal', 'Vitamin C with meals boosts iron absorption'],
  },
  {
    label: '48–60 Months',
    minMonths: 48, maxMonths: 60,
    color: '#795548',
    physical: ['Skips and hops confidently', 'Catches a bounced ball', 'Writes some letters'],
    cognitive: ['Tells stories', 'Counts 10+ objects', 'Understands past/future'],
    feeding: ['Enjoys a wide variety of foods', 'School-age iron needs: 10 mg/day', 'Limit sugary snacks that displace iron-rich foods'],
  },
];

function getCurrentStage(ageMonths: number): MilestoneStage | null {
  return STAGES.find(s => ageMonths >= s.minMonths && ageMonths <= s.maxMonths) ?? null;
}

export default function ParentDevelopmentScreen() {
  const router = useRouter();
  const childAgeMonths = useStore((s) => s.childAgeMonths);
  const childName = useStore((s) => s.childName);

  const stage = childAgeMonths != null ? getCurrentStage(childAgeMonths) : null;

  if (childAgeMonths === null) {
    return (
      <View style={styles.noAge}>
        <Ionicons name="person-outline" size={48} color={colors.textLight} />
        <Text style={styles.noAgeTitle}>Child Age Not Set</Text>
        <Text style={styles.noAgeSub}>Set your child's age in Profile to see development milestones.</Text>
        <View style={{ marginTop: spacing.lg }}>
          <Button title="Go to Profile" onPress={() => router.push('/(parent)/profile')} variant="primary" icon={<Ionicons name="person-outline" size={18} color={colors.white} />} />
        </View>
      </View>
    );
  }

  if (!stage) {
    return (
      <View style={styles.noAge}>
        <Ionicons name="trending-up-outline" size={48} color={colors.textLight} />
        <Text style={styles.noAgeTitle}>Age Outside Range</Text>
        <Text style={styles.noAgeSub}>Milestones are available for children 6–60 months old.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {/* Hero */}
      <View style={[styles.hero, { backgroundColor: stage.color + '15' }]}>
        <View style={[styles.heroIcon, { backgroundColor: stage.color + '25' }]}>
          <Ionicons name="trending-up" size={28} color={stage.color} />
        </View>
        <Text style={styles.heroTitle}>{childName ?? 'Your child'} is</Text>
        <Text style={[styles.heroAge, { color: stage.color }]}>{childAgeMonths} months old</Text>
        <Text style={styles.heroStage}>{stage.label} Stage</Text>
      </View>

      {/* Physical milestones */}
      <MilestoneSection
        icon="body-outline"
        title="Physical Development"
        color={stage.color}
        items={stage.physical}
      />

      {/* Cognitive milestones */}
      <MilestoneSection
        icon="bulb-outline"
        title="Cognitive Development"
        color={stage.color}
        items={stage.cognitive}
      />

      {/* Feeding readiness */}
      <MilestoneSection
        icon="nutrition-outline"
        title="Feeding & Iron Nutrition"
        color={stage.color}
        items={stage.feeding}
      />

      {/* All stages overview */}
      <Text style={styles.sectionTitle}>All Stages</Text>
      <Card>
        {STAGES.map((s, i) => {
          const isCurrent = s.minMonths === stage.minMonths;
          return (
            <View key={s.label} style={[styles.stageRow, i < STAGES.length - 1 && styles.stageRowBorder]}>
              <View style={[styles.stageDot, { backgroundColor: s.color }, isCurrent && styles.stageDotActive]} />
              <Text style={[styles.stageLabel, isCurrent && { color: s.color, fontWeight: '700' }]}>
                {s.label}
              </Text>
              {isCurrent && <View style={[styles.currentBadge, { backgroundColor: s.color + '20' }]}><Text style={[styles.currentBadgeText, { color: s.color }]}>Current</Text></View>}
            </View>
          );
        })}
      </Card>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

function MilestoneSection({ icon, title, color, items }: { icon: any; title: string; color: string; items: string[] }) {
  return (
    <>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Card style={styles.milestoneCard}>
        <View style={styles.milestoneHeader}>
          <View style={[styles.milestoneIcon, { backgroundColor: color + '20' }]}>
            <Ionicons name={icon} size={20} color={color} />
          </View>
          <Text style={[styles.milestoneTitle, { color }]}>{title}</Text>
        </View>
        {items.map((item, i) => (
          <View key={i} style={styles.milestoneItem}>
            <Ionicons name="checkmark-circle-outline" size={16} color={color} />
            <Text style={styles.milestoneText}>{item}</Text>
          </View>
        ))}
      </Card>
    </>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  noAge: { flex: 1, backgroundColor: colors.background, alignItems: 'center', justifyContent: 'center', padding: spacing.xl },
  noAgeTitle: { fontSize: 18, fontWeight: '700', color: colors.textSecondary, marginTop: spacing.md },
  noAgeSub: { fontSize: 14, color: colors.textLight, textAlign: 'center', marginTop: 6, lineHeight: 22 },
  hero: { borderRadius: borderRadius.lg, padding: spacing.xl, alignItems: 'center', marginBottom: spacing.lg },
  heroIcon: { width: 64, height: 64, borderRadius: 32, alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md },
  heroTitle: { fontSize: 16, color: colors.textSecondary, fontWeight: '500' },
  heroAge: { fontSize: 32, fontWeight: '900', marginTop: 4 },
  heroStage: { fontSize: 14, color: colors.textSecondary, marginTop: 4 },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm, marginTop: spacing.md },
  milestoneCard: { marginBottom: spacing.xs },
  milestoneHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.md },
  milestoneIcon: { width: 36, height: 36, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  milestoneTitle: { fontSize: 14, fontWeight: '700' },
  milestoneItem: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm, marginBottom: spacing.sm },
  milestoneText: { flex: 1, fontSize: 14, color: colors.textSecondary, lineHeight: 21 },
  stageRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: spacing.sm, gap: spacing.sm },
  stageRowBorder: { borderBottomWidth: 1, borderBottomColor: colors.border },
  stageDot: { width: 10, height: 10, borderRadius: 5 },
  stageDotActive: { width: 14, height: 14, borderRadius: 7 },
  stageLabel: { flex: 1, fontSize: 14, color: colors.textSecondary },
  currentBadge: { paddingHorizontal: spacing.sm, paddingVertical: 3, borderRadius: borderRadius.pill },
  currentBadgeText: { fontSize: 11, fontWeight: '700' },
});
