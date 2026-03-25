/**
 * app/result.tsx — Display prediction, confidence, nutrition, referral nudge
 */
import React, { memo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../src/shared/theme';
import { severityColorMap, nextScreeningDays } from '../src/shared/theme/colors';
import { Card, ResultBadge, Button } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import { useAnalyticsStore } from '../src/store/analyticsStore';

// ── Nutrition icon map ──
const nutritionIcons: Record<string, keyof typeof Ionicons.glyphMap> = {
  'Non-Anemic': 'leaf-outline',
  'Anemic':     'medical-outline',
  Mild: 'restaurant-outline',
  Moderate: 'medical-outline',
  Severe: 'alert-outline',
};

function ResultScreen() {
  const router = useRouter();
  const result = useStore((s) => s.result);
  const reset = useStore((s) => s.reset);
  const trackEvent = useAnalyticsStore((s) => s.trackEvent);
  const [showAllFoods, setShowAllFoods] = useState(false);

  if (!result) {
    return (
      <View style={styles.emptyContainer}>
        <Ionicons name="alert-circle-outline" size={48} color={colors.textLight} />
        <Text style={styles.emptyText}>No results available</Text>
        <Button
          title="Go Back"
          onPress={() => router.back()}
          variant="outline"
        />
      </View>
    );
  }

  const severityColor =
    severityColorMap[result.prediction] || colors.primary;

  const needsReferral =
    result.prediction === 'Anemic' ||
    result.prediction === 'Moderate' ||
    result.prediction === 'Severe';

  // ── Class Probabilities Bar Chart ──
  const renderProbabilities = () => {
    const entries = Object.entries(result.class_probabilities).sort(
      (a, b) => b[1] - a[1],
    );

    return (
      <Card>
        <Text style={styles.cardTitle}>Class Probabilities</Text>
        {entries.map(([label, prob]) => (
          <View key={label} style={styles.probRow}>
            <Text style={styles.probLabel}>{label}</Text>
            <View style={styles.probBarBg}>
              <View
                style={[
                  styles.probBarFill,
                  {
                    width: `${Math.round(prob * 100)}%`,
                    backgroundColor:
                      severityColorMap[label] || colors.primary,
                  },
                ]}
              />
            </View>
            <Text style={styles.probValue}>{Math.round(prob * 100)}%</Text>
          </View>
        ))}
      </Card>
    );
  };

  // ── Foods List ──
  const visibleFoods = showAllFoods
    ? result.recommended_foods
    : result.recommended_foods.slice(0, 4);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {/* Severity Badge */}
      <ResultBadge
        severity={result.prediction}
        confidence={result.confidence}
      />

      {/* ── Doctor/Referral Nudge Card ── */}
      {needsReferral && (
        <Card
          style={[
            styles.nudgeCard,
            {
              backgroundColor:
                result.prediction === 'Severe'
                  ? colors.severitySevereBg
                  : colors.severityModerateBg,
              borderColor:
                result.prediction === 'Severe'
                  ? colors.severitySevere
                  : colors.severityModerate,
            },
          ]}
        >
          <View style={styles.nudgeRow}>
            <View style={[styles.nudgeIconCircle, { backgroundColor: severityColor }]}>
              <Ionicons name="medkit" size={22} color={colors.white} />
            </View>
            <View style={styles.nudgeTextContainer}>
              <Text style={[styles.nudgeTitle, { color: severityColor }]}>
                {result.prediction === 'Severe'
                  ? 'Urgent: See a Doctor'
                  : result.prediction === 'Anemic'
                  ? 'Follow-up: Clinical Test Recommended'
                  : 'Recommended: See a Doctor'}
              </Text>
              <Text style={styles.nudgeDescription}>
                {result.prediction === 'Severe'
                  ? 'This result indicates severe anemia. Seek medical attention immediately.'
                  : result.prediction === 'Anemic'
                  ? 'Anemia detected from image. A hemoglobin test is needed to determine severity.'
                  : 'This result suggests moderate anemia. Please consult a healthcare provider.'}
              </Text>
            </View>
          </View>
          <Button
            title="View Referral Options"
            onPress={() => {
              trackEvent('referral_opened', { severity: result.prediction });
              router.push('/referral');
            }}
            variant={result.prediction === 'Severe' ? 'danger' : 'primary'}
            icon={<Ionicons name="arrow-forward" size={18} color={colors.white} />}
          />
        </Card>
      )}

      {/* Probabilities */}
      {renderProbabilities()}

      {/* Nutritional Advice */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons
            name={nutritionIcons[result.prediction] || 'nutrition-outline'}
            size={22}
            color={severityColor}
          />
          <Text style={styles.cardTitle}>Nutritional Guidance</Text>
        </View>
        <Text style={styles.adviceText}>{result.nutrition}</Text>
      </Card>

      {/* Recommended Foods */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="fast-food-outline" size={22} color={colors.accent} />
          <Text style={styles.cardTitle}>Recommended Foods</Text>
        </View>
        {visibleFoods.map((food, i) => (
          <View key={i} style={styles.foodRow}>
            <View style={styles.foodBullet}>
              <Ionicons
                name="checkmark-circle"
                size={18}
                color={colors.accentDark}
              />
            </View>
            <Text style={styles.foodText}>{food}</Text>
          </View>
        ))}
        {result.recommended_foods.length > 4 && (
          <TouchableOpacity
            onPress={() => setShowAllFoods(!showAllFoods)}
            style={styles.showMoreBtn}
          >
            <Text style={styles.showMoreText}>
              {showAllFoods
                ? 'Show less'
                : `Show ${result.recommended_foods.length - 4} more`}
            </Text>
            <Ionicons
              name={showAllFoods ? 'chevron-up' : 'chevron-down'}
              size={16}
              color={colors.primary}
            />
          </TouchableOpacity>
        )}
      </Card>

      {/* Next Screening Reminder */}
      {(() => {
        const days = nextScreeningDays[result.prediction] ?? 90;
        const nextDate = new Date();
        nextDate.setDate(nextDate.getDate() + days);
        const dateStr = nextDate.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
        const isUrgent = result.prediction === 'Severe';
        return (
          <Card style={{ backgroundColor: isUrgent ? colors.severitySevereBg : colors.accentLight, borderWidth: 1, borderColor: isUrgent ? colors.severitySevere : colors.accent }}>
            <View style={styles.cardHeader}>
              <Ionicons name="calendar-outline" size={22} color={isUrgent ? colors.severitySevere : colors.accentDark} />
              <Text style={[styles.cardTitle, { color: isUrgent ? colors.severitySevere : colors.accentDark }]}>
                Next Screening Reminder
              </Text>
            </View>
            <Text style={styles.adviceText}>
              {result.prediction === 'Non-Anemic'
                ? 'Great result! Schedule a routine follow-up in 6 months to monitor iron levels.'
                : result.prediction === 'Anemic'
                ? 'Anemia detected. Get a full clinical screen with hemoglobin test to determine severity. Follow up in 30 days.'
                : result.prediction === 'Mild'
                ? 'Mild anemia detected. Re-screen in 3 months to track improvement.'
                : result.prediction === 'Moderate'
                ? 'Moderate anemia detected. Follow up in 4 weeks after starting treatment.'
                : 'Severe anemia — seek immediate care and re-screen in 2 weeks after treatment begins.'}
            </Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 8, backgroundColor: colors.surface, padding: 10, borderRadius: 8 }}>
              <Ionicons name="time-outline" size={16} color={colors.textSecondary} />
              <Text style={{ marginLeft: 6, color: colors.textSecondary, fontWeight: '600' }}>
                Recommended date: {dateStr}
              </Text>
            </View>
          </Card>
        );
      })()}

      {/* Referral Action */}
      <Card
        style={{
          backgroundColor:
            result.prediction === 'Severe'
              ? colors.severitySevereBg
              : colors.surfaceElevated,
        }}
      >
        <View style={styles.cardHeader}>
          <Ionicons
            name="medical-outline"
            size={22}
            color={
              result.prediction === 'Severe'
                ? colors.severitySevere
                : colors.primaryDark
            }
          />
          <Text style={styles.cardTitle}>Medical Referral</Text>
        </View>
        <Text style={styles.referralText}>{result.referral_action}</Text>
      </Card>

      {/* Actions */}
      <View style={styles.actionsRow}>
        <Button
          title="Screen Again"
          onPress={() => {
            reset();
            router.push('/screening');
          }}
          variant="primary"
          style={styles.actionBtn}
          icon={<Ionicons name="refresh" size={18} color={colors.white} />}
        />
        <View style={{ width: spacing.md }} />
        <Button
          title="Learn More"
          onPress={() => router.push('/education')}
          variant="outline"
          style={styles.actionBtn}
          icon={<Ionicons name="book-outline" size={18} color={colors.primary} />}
        />
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
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  emptyText: {
    ...typography.body,
    color: colors.textLight,
    marginVertical: spacing.md,
  },
  // ── Nudge Card ──
  nudgeCard: {
    borderWidth: 2,
    marginBottom: spacing.md,
  },
  nudgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  nudgeIconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  nudgeTextContainer: {
    flex: 1,
  },
  nudgeTitle: {
    ...typography.subtitle,
    marginBottom: 2,
  },
  nudgeDescription: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  // ── Cards ──
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  cardTitle: {
    ...typography.subtitle,
    color: colors.text,
    marginLeft: spacing.sm,
  },
  adviceText: {
    ...typography.body,
    color: colors.textSecondary,
    lineHeight: 24,
  },
  // ── Probabilities ──
  probRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  probLabel: {
    ...typography.caption,
    color: colors.textSecondary,
    width: 90,
  },
  probBarBg: {
    flex: 1,
    height: 10,
    backgroundColor: colors.surfaceElevated,
    borderRadius: 5,
    overflow: 'hidden',
    marginHorizontal: spacing.sm,
  },
  probBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  probValue: {
    ...typography.captionBold,
    color: colors.text,
    width: 40,
    textAlign: 'right',
  },
  // ── Foods ──
  foodRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  foodBullet: {
    marginRight: spacing.sm,
  },
  foodText: {
    ...typography.body,
    color: colors.text,
    flex: 1,
  },
  showMoreBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
  },
  showMoreText: {
    ...typography.captionBold,
    color: colors.primary,
    marginRight: spacing.xs,
  },
  // ── Referral ──
  referralText: {
    ...typography.body,
    color: colors.text,
    lineHeight: 24,
  },
  // ── Actions ──
  actionsRow: {
    flexDirection: 'row',
    marginTop: spacing.md,
  },
  actionBtn: {
    flex: 1,
  },
});

export default memo(ResultScreen);
