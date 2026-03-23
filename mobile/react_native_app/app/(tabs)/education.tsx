/**
 * app/education.tsx — Educational content about anemia
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

// ── Education content data ──
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
    title: 'What is Anemia?',
    icon: 'information-circle-outline',
    iconColor: colors.primary,
    content: [
      'Anemia is a condition where the body lacks enough healthy red blood cells to carry adequate oxygen to tissues.',
      'It can be caused by iron deficiency, vitamin B12 deficiency, chronic diseases, or genetic conditions.',
      'The World Health Organization estimates that anemia affects approximately 1.62 billion people globally.',
      'Early detection and proper nutrition can help manage and prevent anemia effectively.',
    ],
  },
  {
    id: 'symptoms',
    title: 'Common Symptoms',
    icon: 'fitness-outline',
    iconColor: colors.secondaryDark,
    content: [
      '😮‍💨  Fatigue and weakness',
      '😵‍💫  Dizziness or lightheadedness',
      '💛  Pale or yellowish skin',
      '💓  Irregular or fast heartbeat',
      '🫁  Shortness of breath',
      '🥶  Cold hands and feet',
      '🤕  Headaches',
      '💅  Brittle nails',
    ],
  },
  {
    id: 'iron',
    title: 'Iron-Rich Foods',
    icon: 'leaf-outline',
    iconColor: colors.accentDark,
    content: [
      '🥩  Lean red meat and liver',
      '🐔  Poultry (chicken, turkey)',
      '🐟  Fish and shellfish (oysters, clams)',
      '🥬  Dark leafy greens (spinach, kale)',
      '🫘  Legumes (lentils, beans, chickpeas)',
      '🥜  Nuts and seeds (pumpkin seeds)',
      '🥣  Fortified cereals and bread',
      '🫛  Tofu and tempeh',
    ],
  },
  {
    id: 'vitc',
    title: 'Vitamin C Foods',
    icon: 'sunny-outline',
    iconColor: '#FFB74D',
    content: [
      '🍊  Citrus fruits (oranges, lemons, grapefruit)',
      '🫑  Bell peppers (red and green)',
      '🍓  Strawberries and kiwi',
      '🥦  Broccoli and Brussels sprouts',
      '🍅  Tomatoes and tomato juice',
      '🥔  Potatoes (with skin)',
      '',
      '💡 Vitamin C enhances iron absorption when consumed with iron-rich foods.',
    ],
  },
  {
    id: 'prevention',
    title: 'Prevention Tips',
    icon: 'shield-checkmark-outline',
    iconColor: '#7E57C2',
    content: [
      'Eat a balanced diet rich in iron and vitamins',
      'Pair iron-rich foods with vitamin C sources',
      'Avoid tea or coffee during meals (they reduce iron absorption)',
      'Get regular blood tests, especially during pregnancy',
      'Consider iron supplements if recommended by a doctor',
      'Cook in iron cookware to boost dietary iron intake',
    ],
  },
];

// ── Expandable Section Component ──
const ExpandableSection: React.FC<{ section: EducationSection }> = memo(
  ({ section }) => {
    const [expanded, setExpanded] = useState(section.id === 'what');

    return (
      <Card style={styles.sectionCard}>
        <TouchableOpacity
          style={styles.sectionHeader}
          onPress={() => setExpanded(!expanded)}
          activeOpacity={0.7}
        >
          <View style={[styles.sectionIcon, { backgroundColor: section.iconColor + '20' }]}>
            <Ionicons name={section.icon} size={22} color={section.iconColor} />
          </View>
          <Text style={styles.sectionTitle}>{section.title}</Text>
          <Ionicons
            name={expanded ? 'chevron-up' : 'chevron-down'}
            size={20}
            color={colors.textLight}
          />
        </TouchableOpacity>

        {expanded && (
          <View style={styles.sectionContent}>
            {section.content.map((item, i) => (
              <Text key={i} style={[styles.contentText, !item && { height: spacing.sm }]}>
                {item}
              </Text>
            ))}
          </View>
        )}
      </Card>
    );
  },
);

// ── Main Screen ──
export default function EducationScreen() {
  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {/* Hero */}
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="school-outline" size={32} color={colors.primary} />
        </View>
        <Text style={styles.heroTitle}>Understanding Anemia</Text>
        <Text style={styles.heroSubtitle}>
          Learn about symptoms, prevention, and the foods that can help.
        </Text>
      </View>

      {/* Sections */}
      {EDUCATION_DATA.map((section) => (
        <ExpandableSection key={section.id} section={section} />
      ))}

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Ionicons name="information-circle" size={16} color={colors.textLight} />
        <Text style={styles.disclaimerText}>
          This information is for educational purposes only and does not replace
          professional medical advice.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing.xxl,
  },
  // ── Hero ──
  hero: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  heroIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  heroTitle: {
    ...typography.title,
    color: colors.text,
    textAlign: 'center',
  },
  heroSubtitle: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing.xs,
  },
  // ── Sections ──
  sectionCard: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  sectionIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  sectionTitle: {
    ...typography.subtitle,
    color: colors.text,
    flex: 1,
  },
  sectionContent: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  contentText: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
    lineHeight: 24,
  },
  // ── Disclaimer ──
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: spacing.md,
    marginTop: spacing.md,
  },
  disclaimerText: {
    ...typography.caption,
    color: colors.textLight,
    marginLeft: spacing.sm,
    flex: 1,
    fontStyle: 'italic',
  },
});
