/**
 * app/(chw)/followup.tsx — Follow-up scheduling screen
 *
 * Displays upcoming follow-ups computed from screening history.
 * Next screening date is calculated based on severity:
 *   Non-Anemic → 180 days, Mild → 90 days,
 *   Moderate → 30 days, Severe → 14 days
 */
import React, { useMemo } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../../src/shared/theme';
import { nextScreeningDays, severityColorMap, severityBgMap } from '../../src/shared/theme/colors';
import { useStore } from '../../src/store/useStore';
import { Card } from '../../src/shared/components';

interface FollowUp {
  id: string;
  patientName: string;
  prediction: string;
  lastScreeningDate: Date;
  nextScreeningDate: Date;
  daysUntil: number;
  isOverdue: boolean;
}

export default function FollowupScreen() {
  const history = useStore((s) => s.history);

  const followups = useMemo((): FollowUp[] => {
    const now = new Date();
    return history
      .filter((r) => r.prediction !== 'Non-Anemic') // Only track anemic patients
      .map((r) => {
        const lastDate = new Date(r.date);
        const intervalDays = nextScreeningDays[r.prediction] ?? 90;
        const nextDate = new Date(lastDate.getTime() + intervalDays * 24 * 60 * 60 * 1000);
        const daysUntil = Math.ceil((nextDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));

        return {
          id: r.id,
          patientName: r.patientName || 'Unknown Patient',
          prediction: r.prediction,
          lastScreeningDate: lastDate,
          nextScreeningDate: nextDate,
          daysUntil,
          isOverdue: daysUntil < 0,
        };
      })
      .sort((a, b) => a.daysUntil - b.daysUntil); // Overdue first, then soonest
  }, [history]);

  const overdueCount = followups.filter((f) => f.isOverdue).length;
  const upcomingCount = followups.filter((f) => !f.isOverdue).length;

  if (followups.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <View style={styles.emptyIconCircle}>
          <Ionicons name="calendar-outline" size={40} color={colors.primary} />
        </View>
        <Text style={styles.emptyTitle}>No Pending Follow-Ups</Text>
        <Text style={styles.emptySub}>
          Follow-up reminders will appear here when patients with anemia are due for re-screening.
        </Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Summary header */}
      <View style={styles.summaryRow}>
        {overdueCount > 0 && (
          <View style={[styles.summaryPill, { backgroundColor: colors.errorBg }]}>
            <Ionicons name="alert-circle" size={14} color={colors.error} />
            <Text style={[styles.summaryText, { color: colors.error }]}>
              {overdueCount} overdue
            </Text>
          </View>
        )}
        <View style={[styles.summaryPill, { backgroundColor: colors.primaryLight + '40' }]}>
          <Ionicons name="time-outline" size={14} color={colors.primary} />
          <Text style={[styles.summaryText, { color: colors.primary }]}>
            {upcomingCount} upcoming
          </Text>
        </View>
      </View>

      {/* Follow-up cards */}
      {followups.map((fu) => {
        const severityColor = severityColorMap[fu.prediction] ?? colors.textSecondary;
        const severityBg = severityBgMap[fu.prediction] ?? colors.surfaceElevated;

        return (
          <Card key={fu.id} style={fu.isOverdue ? styles.overdueCard : undefined}>
            <View style={styles.cardRow}>
              <View style={[styles.severityDot, { backgroundColor: severityColor }]} />
              <View style={styles.cardInfo}>
                <Text style={styles.patientName}>{fu.patientName}</Text>
                <View style={styles.metaRow}>
                  <View style={[styles.severityBadge, { backgroundColor: severityBg }]}>
                    <Text style={[styles.severityText, { color: severityColor }]}>
                      {fu.prediction}
                    </Text>
                  </View>
                  <Text style={styles.dateText}>
                    Last: {fu.lastScreeningDate.toLocaleDateString()}
                  </Text>
                </View>
              </View>
              <View style={styles.daysContainer}>
                {fu.isOverdue ? (
                  <>
                    <Text style={styles.daysOverdue}>{Math.abs(fu.daysUntil)}d</Text>
                    <Text style={styles.daysLabel}>overdue</Text>
                  </>
                ) : fu.daysUntil === 0 ? (
                  <>
                    <Text style={styles.daysToday}>Today</Text>
                    <Text style={styles.daysLabel}>due</Text>
                  </>
                ) : (
                  <>
                    <Text style={styles.daysCount}>{fu.daysUntil}d</Text>
                    <Text style={styles.daysLabel}>left</Text>
                  </>
                )}
              </View>
            </View>
            <View style={styles.nextDateRow}>
              <Ionicons name="calendar-outline" size={13} color={colors.textLight} />
              <Text style={styles.nextDateText}>
                Next screening: {fu.nextScreeningDate.toLocaleDateString()}
              </Text>
            </View>
          </Card>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xxl,
  },
  // ── Empty state ──
  emptyContainer: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  emptyIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primaryLight + '40',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.text,
    textAlign: 'center',
  },
  emptySub: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: spacing.sm,
    textAlign: 'center',
    lineHeight: 22,
  },
  // ── Summary ──
  summaryRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  summaryPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  summaryText: {
    ...typography.captionBold,
  },
  // ── Cards ──
  overdueCard: {
    borderWidth: 1.5,
    borderColor: colors.error,
    backgroundColor: colors.errorBg,
  },
  cardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  severityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  cardInfo: {
    flex: 1,
  },
  patientName: {
    ...typography.bodyBold,
    color: colors.text,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: 4,
  },
  severityBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  severityText: {
    ...typography.captionBold,
  },
  dateText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  daysContainer: {
    alignItems: 'center',
    minWidth: 50,
  },
  daysOverdue: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.error,
  },
  daysToday: {
    fontSize: 14,
    fontWeight: '800',
    color: colors.warning,
  },
  daysCount: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.primary,
  },
  daysLabel: {
    ...typography.caption,
    color: colors.textLight,
    fontSize: 11,
  },
  nextDateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: spacing.sm,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.borderLight,
  },
  nextDateText: {
    ...typography.caption,
    color: colors.textLight,
  },
});
