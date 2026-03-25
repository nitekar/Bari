/**
 * app/(tabs)/history.tsx — Screening history + analytics dashboard
 */
import React, { useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  FlatList,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Rect, Text as SvgText } from 'react-native-svg';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { severityColorMap, severityBgMap, nextScreeningDays } from '../../src/shared/theme/colors';
import { Card, ResultBadge } from '../../src/shared/components';
import { useStore, type ScreeningRecord } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';

const SEVERITY_ORDER = ['Non-Anemic', 'Mild', 'Moderate', 'Severe'];
const CHART_WIDTH = 280;
const CHART_HEIGHT = 120;
const BAR_GAP = 12;

export default function HistoryScreen() {
  const { t } = useTranslation();
  const history = useStore((s) => s.history);

  const stats = useMemo(() => {
    if (!history.length) return null;

    const dist: Record<string, number> = {};
    let totalConf = 0;
    history.forEach((r) => {
      dist[r.prediction] = (dist[r.prediction] || 0) + 1;
      totalConf += r.confidence;
    });

    const mostCommon = Object.entries(dist).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—';
    const lastDate = history[0]?.date
      ? new Date(history[0].date).toLocaleDateString()
      : '—';

    return {
      total: history.length,
      avgConf: Math.round((totalConf / history.length) * 100),
      mostCommon,
      lastDate,
      dist,
    };
  }, [history]);

  // ── Bar chart for severity breakdown ──────────────────────────────────────
  const renderChart = () => {
    if (!stats) return null;
    const maxVal = Math.max(...SEVERITY_ORDER.map((k) => stats.dist[k] ?? 0), 1);
    const barW = (CHART_WIDTH - BAR_GAP * (SEVERITY_ORDER.length + 1)) / SEVERITY_ORDER.length;

    return (
      <Card style={styles.chartCard}>
        <Text style={styles.cardTitle}>{t.history.severityBreakdown}</Text>
        <Svg width={CHART_WIDTH} height={CHART_HEIGHT + 24}>
          {SEVERITY_ORDER.map((label, i) => {
            const val = stats.dist[label] ?? 0;
            const barH = Math.max((val / maxVal) * CHART_HEIGHT, val > 0 ? 8 : 0);
            const x = BAR_GAP + i * (barW + BAR_GAP);
            const y = CHART_HEIGHT - barH;
            const barColor = severityColorMap[label] ?? colors.primary;
            return (
              <React.Fragment key={label}>
                <Rect
                  x={x}
                  y={y}
                  width={barW}
                  height={barH}
                  fill={barColor}
                  rx={6}
                />
                <SvgText
                  x={x + barW / 2}
                  y={CHART_HEIGHT + 16}
                  fontSize={9}
                  fill={colors.textSecondary}
                  textAnchor="middle"
                >
                  {label.replace('-', '\n')}
                </SvgText>
                {val > 0 && (
                  <SvgText
                    x={x + barW / 2}
                    y={y - 4}
                    fontSize={10}
                    fill={barColor}
                    textAnchor="middle"
                    fontWeight="700"
                  >
                    {val}
                  </SvgText>
                )}
              </React.Fragment>
            );
          })}
        </Svg>
      </Card>
    );
  };

  // ── Summary stat cards ────────────────────────────────────────────────────
  const renderStats = () => {
    if (!stats) return null;
    return (
      <View style={styles.statsRow}>
        <StatChip label={t.history.totalScreenings} value={String(stats.total)} color={colors.primary} />
        <StatChip label={t.history.avgConfidence} value={`${stats.avgConf}%`} color={colors.severityNormal} />
        <StatChip label={t.history.mostCommon} value={stats.mostCommon} color={severityColorMap[stats.mostCommon] ?? colors.primary} />
        <StatChip label={t.history.lastScreening} value={stats.lastDate} color={colors.secondaryDark} />
      </View>
    );
  };

  // ── History list item ─────────────────────────────────────────────────────
  const renderItem = ({ item }: { item: ScreeningRecord }) => {
    const severityColor = severityColorMap[item.prediction] ?? colors.primary;
    const date = new Date(item.date);
    const isToday = date.toDateString() === new Date().toDateString();
    const dateStr = isToday ? t.history.today : date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return (
      <View style={styles.historyItem}>
        {/* Severity accent bar */}
        <View style={[styles.severityBar, { backgroundColor: severityColor }]} />
        <View style={styles.historyBody}>
          {/* Top row: patient name + badge */}
          <View style={styles.historyTopRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.patientName} numberOfLines={1}>
                {item.patientName || 'Unknown Patient'}
              </Text>
              {item.patientLocation ? (
                <View style={styles.locationRow}>
                  <Ionicons name="location-outline" size={12} color={colors.textLight} />
                  <Text style={styles.locationText}>{item.patientLocation}</Text>
                </View>
              ) : null}
            </View>
            <View style={[styles.severityPill, { backgroundColor: severityColor + '20', borderColor: severityColor }]}>
              <Text style={[styles.severityPillText, { color: severityColor }]}>{item.prediction}</Text>
            </View>
          </View>

          {/* Bottom row: age + date + confidence */}
          <View style={styles.historyBottomRow}>
            {item.age != null && (
              <View style={styles.metaChip}>
                <Ionicons name="person-outline" size={11} color={colors.textSecondary} />
                <Text style={styles.metaText}>{item.age} yrs</Text>
              </View>
            )}
            <View style={styles.metaChip}>
              <Ionicons name="calendar-outline" size={11} color={colors.textSecondary} />
              <Text style={styles.metaText}>{dateStr} · {timeStr}</Text>
            </View>
            <View style={[styles.confBadge, { backgroundColor: severityColor }]}>
              <Text style={styles.confText}>{Math.round(item.confidence * 100)}%</Text>
            </View>
          </View>

          {/* Next screening reminder */}
          {(() => {
            const days = nextScreeningDays[item.prediction] ?? 90;
            const nextDate = new Date(item.date);
            nextDate.setDate(nextDate.getDate() + days);
            const isPast = nextDate < new Date();
            const nextDateStr = nextDate.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
            return (
              <View style={[styles.reminderRow, { backgroundColor: isPast ? colors.severitySevereBg : colors.accentLight }]}>
                <Ionicons name="alarm-outline" size={12} color={isPast ? colors.severitySevere : colors.accentDark} />
                <Text style={[styles.reminderText, { color: isPast ? colors.severitySevere : colors.accentDark }]}>
                  {isPast ? 'Follow-up overdue — ' : 'Next screening: '}
                  <Text style={{ fontWeight: '700' }}>{nextDateStr}</Text>
                </Text>
              </View>
            );
          })()}
        </View>
      </View>
    );
  };

  // ── Empty state ───────────────────────────────────────────────────────────
  if (!history.length) {
    return (
      <View style={styles.empty}>
        <View style={styles.emptyIcon}>
          <Ionicons name="time-outline" size={48} color={colors.primary} />
        </View>
        <Text style={styles.emptyTitle}>{t.history.noHistory}</Text>
        <Text style={styles.emptySub}>{t.history.noHistorySub}</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {renderStats()}
      {renderChart()}

      <Text style={styles.sectionTitle}>{t.history.recentActivity}</Text>
      <FlatList
        data={history}
        keyExtractor={(r) => r.id}
        renderItem={renderItem}
        scrollEnabled={false}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
      />

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function StatChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <View style={[styles.statChip, { borderTopColor: color, borderTopWidth: 3 }]}>
      <Text style={[styles.statValue, { color }]} numberOfLines={1}>{value}</Text>
      <Text style={styles.statLabel} numberOfLines={2}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  // ── Stats row ──
  statsRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: spacing.lg,
  },
  statChip: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    alignItems: 'center',
    ...shadows.sm,
  },
  statValue: {
    fontSize: 15,
    fontWeight: '800',
    marginBottom: 2,
  },
  statLabel: {
    fontSize: 9,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 12,
  },
  // ── Chart ──
  chartCard: { alignItems: 'center', marginBottom: spacing.lg },
  cardTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.text,
    marginBottom: spacing.md,
    alignSelf: 'flex-start',
  },
  // ── Section title ──
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
    marginBottom: spacing.md,
  },
  // ── History item ──
  historyItem: {
    flexDirection: 'row',
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    ...shadows.sm,
  },
  severityBar: { width: 5 },
  historyBody: { flex: 1, padding: spacing.md },
  historyTopRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 8 },
  patientName: { fontSize: 15, fontWeight: '700', color: colors.text },
  locationRow: { flexDirection: 'row', alignItems: 'center', gap: 2, marginTop: 2 },
  locationText: { fontSize: 11, color: colors.textLight },
  severityPill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
    borderWidth: 1,
    marginLeft: 8,
    alignSelf: 'flex-start',
  },
  severityPillText: { fontSize: 11, fontWeight: '700' },
  historyBottomRow: { flexDirection: 'row', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
  metaChip: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  metaText: { fontSize: 11, color: colors.textSecondary, textTransform: 'capitalize' },
  confBadge: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: 12, marginLeft: 'auto' as any },
  confText: { fontSize: 11, fontWeight: '700', color: colors.white },
  reminderRow: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    marginTop: 8, paddingHorizontal: 8, paddingVertical: 5,
    borderRadius: 7,
  },
  reminderText: { fontSize: 11, flex: 1 },
  // ── Empty ──
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xxl,
    backgroundColor: colors.background,
  },
  emptyIcon: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: colors.text, textAlign: 'center', marginBottom: spacing.sm },
  emptySub: { fontSize: 14, color: colors.textSecondary, textAlign: 'center', lineHeight: 20 },
});
