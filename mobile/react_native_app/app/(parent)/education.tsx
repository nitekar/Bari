/**
 * app/(parent)/education.tsx — Educational content about anemia
 * Copy of (tabs)/education.tsx for the parent route group.
 */
import React, { memo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';

interface EducationSection {
  id: string;
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  iconColor: string;
  content: string[];
}

const EDUCATION_DATA: EducationSection[] = [
  {
    id: 'what',
    title: 'What is Anemia in Children?',
    icon: 'information-circle-outline',
    iconColor: colors.primary,
    content: [
      "Anemia in children occurs when the blood does not have enough healthy red blood cells to carry oxygen to the body's organs and muscles.",
      'In children under 5, the most common cause is iron deficiency — a critical nutrient required for brain development, muscle growth, and immune function.',
      'The WHO estimates that 42% of children under 5 globally are affected by anemia, with the highest burden in Sub-Saharan Africa.',
      'Untreated anemia during early childhood can permanently impair cognitive development, physical growth, and school performance.',
      'Early detection through conjunctiva screening and prompt nutritional support can prevent long-term harm.',
    ],
  },
  {
    id: 'growth',
    title: 'How Anemia Affects Child Growth',
    icon: 'body-outline',
    iconColor: colors.secondaryDark,
    content: [
      'Brain development: Iron is essential for myelination — the process of building the nerve fiber insulation needed for fast thinking and learning. Iron deficiency during the first 1,000 days of life leads to irreversible cognitive delays.',
      'Height and weight: Anemic children have reduced appetite and poor nutrient absorption, leading to stunted growth and underweight status.',
      'Muscle strength: Iron deficiency reduces oxygen delivery to muscles, causing fatigue, weakness, and poor physical activity tolerance.',
      'Immune defense: Iron supports immune cell production. Anemic children have weaker immunity and are more vulnerable to infections such as pneumonia and diarrhea.',
      'Energy and mood: Chronic fatigue from anemia causes irritability, poor concentration, and reduced engagement in learning activities.',
    ],
  },
  {
    id: 'iron',
    title: 'Iron-Rich Foods for Children',
    icon: 'leaf-outline',
    iconColor: colors.accentDark,
    content: [
      'Liver and organ meats — the richest source of heme iron; even small amounts weekly significantly raise hemoglobin levels in young children.',
      'Dark leafy greens (spinach, amaranth, moringa) — locally available sources of non-heme iron.',
      'Legumes (beans, lentils, soybeans) — a primary iron source in plant-based diets.',
      'Eggs — provide both iron and protein, supporting red blood cell production and muscle growth.',
      'Small dried fish — widely available, affordable, and rich in both iron and zinc.',
      'Fortified porridges — designed specifically for infants and toddlers to meet daily iron requirements.',
    ],
  },
  {
    id: 'absorption',
    title: 'Maximising Iron Absorption',
    icon: 'sunny-outline',
    iconColor: '#FFB74D',
    content: [
      'Always pair iron-rich foods with Vitamin C sources (orange, tomato, guava, passion fruit). Vitamin C converts non-heme iron into a form the body absorbs 3x more efficiently.',
      "Avoid giving tea, coffee, or cow's milk during or directly after iron-rich meals — tannins and calcium block iron absorption significantly.",
      'Add lemon juice or tomato to bean dishes at serving time to boost iron uptake without changing the meal.',
      'Ferment or soak grains and legumes overnight before cooking — this reduces phytates, the compounds that bind iron and prevent absorption.',
      'If a child is diagnosed with moderate or severe anemia, a doctor may prescribe iron drops or syrup.',
    ],
  },
  {
    id: 'feeding',
    title: 'Feeding Guidelines by Age',
    icon: 'restaurant-outline',
    iconColor: '#5B9EC9',
    content: [
      "0-6 months: Exclusive breastfeeding provides all iron needs. Mother's diet should include iron-rich foods.",
      '6-12 months: Introduce iron-rich first foods from 6 months. Start with iron-fortified porridge, mashed beans, or pureed liver.',
      '1-3 years: Serve 3 meals and 2 nutritious snacks daily. Include animal-source food at least 3-4 times per week.',
      '4-5 years: Ensure varied diet across all food groups. Make meals appealing, colorful, and served in small portions frequently.',
      "All ages: Never give cow's milk as a main drink before 12 months — it is low in iron and inhibits absorption.",
    ],
  },
  {
    id: 'prevention',
    title: 'Prevention at Home',
    icon: 'shield-checkmark-outline',
    iconColor: '#7E57C2',
    content: [
      'Promote kitchen gardens growing moringa, amaranth, and other iron-rich vegetables.',
      'Encourage use of iron cooking pots — cooking acidic foods in cast iron pots can meaningfully increase dietary iron.',
      'Refer all children with Moderate or Severe screening results immediately to a health facility for testing and treatment.',
      'Educate yourself that good nutrition must continue even after a child recovers — anemia recurrence is common without sustained dietary change.',
      'Schedule follow-up screenings as recommended: every 90 days for Mild, every 30 days for Moderate.',
    ],
  },
];

const ExpandableSection: React.FC<{ section: EducationSection }> = memo(({ section }) => {
  const [expanded, setExpanded] = useState(section.id === 'what');

  return (
    <Card style={styles.sectionCard}>
      <TouchableOpacity style={styles.sectionHeader} onPress={() => setExpanded(!expanded)} activeOpacity={0.7}>
        <View style={[styles.sectionIcon, { backgroundColor: section.iconColor + '20' }]}>
          <Ionicons name={section.icon} size={22} color={section.iconColor} />
        </View>
        <Text style={styles.sectionTitle}>{section.title}</Text>
        <Ionicons name={expanded ? 'chevron-up' : 'chevron-down'} size={20} color={colors.textLight} />
      </TouchableOpacity>
      {expanded && (
        <View style={styles.sectionContent}>
          {section.content.map((item, i) => (
            <Text key={i} style={[styles.contentText, !item && { height: spacing.sm }]}>{item}</Text>
          ))}
        </View>
      )}
    </Card>
  );
});

export default function EducationScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="school-outline" size={32} color={colors.primary} />
        </View>
        <Text style={styles.heroTitle}>Nutrition & Child Growth</Text>
        <Text style={styles.heroSubtitle}>
          How iron deficiency affects child development and what to do about it.
        </Text>
      </View>
      {EDUCATION_DATA.map((section) => (
        <ExpandableSection key={section.id} section={section} />
      ))}
      <View style={styles.disclaimer}>
        <Ionicons name="information-circle" size={16} color={colors.textLight} />
        <Text style={styles.disclaimerText}>
          This information is for educational purposes only and does not replace professional medical advice.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },
  hero: { alignItems: 'center', marginBottom: spacing.xl },
  heroIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.primary + '20', alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md },
  heroTitle: { ...typography.title, color: colors.text, textAlign: 'center' },
  heroSubtitle: { ...typography.body, color: colors.textSecondary, textAlign: 'center', marginTop: spacing.xs },
  sectionCard: { paddingVertical: spacing.md, paddingHorizontal: spacing.md },
  sectionHeader: { flexDirection: 'row', alignItems: 'center' },
  sectionIcon: { width: 40, height: 40, borderRadius: 12, alignItems: 'center', justifyContent: 'center', marginRight: spacing.md },
  sectionTitle: { ...typography.subtitle, color: colors.text, flex: 1 },
  sectionContent: { marginTop: spacing.md, paddingTop: spacing.md, borderTopWidth: 1, borderTopColor: colors.border },
  contentText: { ...typography.body, color: colors.textSecondary, marginBottom: spacing.sm, lineHeight: 24 },
  disclaimer: { flexDirection: 'row', alignItems: 'flex-start', padding: spacing.md, marginTop: spacing.md },
  disclaimerText: { ...typography.caption, color: colors.textLight, marginLeft: spacing.sm, flex: 1, fontStyle: 'italic' },
});
