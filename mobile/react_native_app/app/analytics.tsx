/**
 * app/analytics.tsx — Local-only analytics dashboard
 *
 * Layout:
 * 1. Summary cards (total screenings, avg confidence, most common, last date)
 * 2. Severity distribution bar chart (pure react-native-svg)
 * 3. Screenings over time trend line (pure react-native-svg)
 * 4. Recent screening history
 * 5. Usage insights
 *
 * ⚠️ No Victory Native, Recharts, or any charting library — pure react-native-svg.
 */
import React, { useMemo } from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Rect, Text as SvgText, Line, Polyline, Circle } from 'react-native-svg';
import { colors, typography, spacing, borderRadius, shadows } from '../src/shared/theme';
import { severityColorMap } from '../src/shared/theme/colors';
import { Card, ResultBadge } from '../src/shared/components';
import { useAnalyticsStore } from '../src/app/store/analyticsStore';
import { useStore } from '../src/app/store/useStore';

// ── Constants ──
const CHART_WIDTH = 300;
const CHART_HEIGHT = 180;
const BAR_CHART_PADDING = 40;
const SEVERITY_ORDER = ['Non-Anemic', 'Mild', 'Moderate', 'Severe'];

// ── Summary Card Component ──
function SummaryCard({
  icon,
  iconColor,
  label,
  value,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  iconColor: string;
  label: string;
  value: string;
}) {
  return (
    <View style={styles.summaryCard}>
      <View style={[styles.summaryIconCircle, { backgroundColor: iconColor + '20' }]}>
        <Ionicons name={icon} size={20} color={iconColor} />
      </View>
      <Text style={styles.summaryValue}>{value}</Text>
      <Text style={styles.summaryLabel}>{label}</Text>
    </View>
  );
}

// ── Bar Chart (Severity Distribution) ──
function SeverityBarChart({ distribution }: { distribution: Record<string, number> }) {
  const maxValue = Math.max(...Object.values(distribution), 1);
  const barWidth = (CHART_WIDTH - BAR_CHART_PADDING * 2) / SEVERITY_ORDER.length - 8;

  return (
    <Card>
      <View style={styles.cardHeader}>
        <Ionicons name="bar-chart-outline" size={20} color={colors.primary} />
        <Text style={styles.cardTitle}>Severity Distribution</Text>
      </View>
      <View style={styles.chartContainer}>
        <Svg width={CHART_WIDTH} height={CHART_HEIGHT} viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}>
          {/* Y-axis line */}
          <Line
            x1={BAR_CHART_PADDING}
            y1={10}
            x2={BAR_CHART_PADDING}
            y2={CHART_HEIGHT - 30}
            stroke={colors.border}
            strokeWidth={1}
          />
          {/* X-axis line */}
          <Line
            x1={BAR_CHART_PADDING}
            y1={CHART_HEIGHT - 30}
            x2={CHART_WIDTH - 10}
            y2={CHART_HEIGHT - 30}
            stroke={colors.border}
            strokeWidth={1}
          />

          {SEVERITY_ORDER.map((label, i) => {
            const count = distribution[label] || 0;
            const barHeight = maxValue > 0 ? (count / maxValue) * (CHART_HEIGHT - 50) : 0;
            const x = BAR_CHART_PADDING + 8 + i * (barWidth + 8);
            const y = CHART_HEIGHT - 30 - barHeight;
            const barColor = severityColorMap[label] || colors.primary;

            return (
              <React.Fragment key={label}>
                {/* Bar */}
                <Rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx={4}
                  fill={barColor}
                  opacity={0.85}
                />
                {/* Value above bar */}
                {count > 0 && (
                  <SvgText
                    x={x + barWidth / 2}
                    y={y - 5}
                    fontSize={11}
                    fontWeight="600"
                    fill={colors.text}
                    textAnchor="middle"
                  >
                    {count}
                  </SvgText>
                )}
                {/* Label below bar */}
                <SvgText
                  x={x + barWidth / 2}
                  y={CHART_HEIGHT - 12}
                  fontSize={9}
                  fill={colors.textSecondary}
                  textAnchor="middle"
                >
                  {label.length > 8 ? label.slice(0, 7) + '…' : label}
                </SvgText>
              </React.Fragment>
            );
          })}
        </Svg>
      </View>
    </Card>
  );
}

// ── Trend Line Chart (Screenings Over Time) ──
function TrendLineChart({ screeningsPerDay }: { screeningsPerDay: Record<string, number> }) {
  const sortedDays = Object.keys(screeningsPerDay).sort();
  const lastDays = sortedDays.slice(-14); // last 14 days

  if (lastDays.length < 2) {
    return (
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="trending-up-outline" size={20} color={colors.accent} />
          <Text style={styles.cardTitle}>Screenings Trend</Text>
        </View>
        <Text style={styles.emptyChartText}>
          Need at least 2 days of data to show trends.
        </Text>
      </Card>
    );
  }

  const values = lastDays.map((d) => screeningsPerDay[d]);
  const maxVal = Math.max(...values, 1);
  const padding = 30;
  const chartW = CHART_WIDTH;
  const chartH = 140;
  const graphW = chartW - padding * 2;
  const graphH = chartH - padding * 2;

  const points = lastDays.map((_, i) => {
    const x = padding + (i / (lastDays.length - 1)) * graphW;
    const y = padding + graphH - (values[i] / maxVal) * graphH;
    return { x, y };
  });

  const polylinePoints = points.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <Card>
      <View style={styles.cardHeader}>
        <Ionicons name="trending-up-outline" size={20} color={colors.accent} />
        <Text style={styles.cardTitle}>Screenings Trend</Text>
      </View>
      <View style={styles.chartContainer}>
        <Svg width={chartW} height={chartH} viewBox={`0 0 ${chartW} ${chartH}`}>
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
            const y = padding + graphH - frac * graphH;
            return (
              <Line
                key={frac}
                x1={padding}
                y1={y}
                x2={chartW - padding}
                y2={y}
                stroke={colors.border}
                strokeWidth={0.5}
                strokeDasharray="4,4"
              />
            );
          })}

          {/* Trend line */}
          <Polyline
            points={polylinePoints}
            fill="none"
            stroke={colors.accentDark}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Data points */}
          {points.map((p, i) => (
            <Circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={4}
              fill={colors.white}
              stroke={colors.accentDark}
              strokeWidth={2}
            />
          ))}

          {/* X-axis labels (first and last day) */}
          <SvgText
            x={padding}
            y={chartH - 5}
            fontSize={9}
            fill={colors.textLight}
            textAnchor="start"
          >
            {lastDays[0].slice(5)}
          </SvgText>
          <SvgText
            x={chartW - padding}
            y={chartH - 5}
            fontSize={9}
            fill={colors.textLight}
            textAnchor="end"
          >
            {lastDays[lastDays.length - 1].slice(5)}
          </SvgText>
        </Svg>
      </View>
    </Card>
  );
}

// ── Main Screen ──
export default function AnalyticsDashboardScreen() {
  const aggregates = useAnalyticsStore((s) => s.getAggregates());
  const history = useStore((s) => s.history);

  const recentHistory = useMemo(() => history.slice(0, 10), [history]);

  const lastDateText = aggregates.lastScreeningDate
    ? new Date(aggregates.lastScreeningDate).toLocaleDateString()
    : 'Never';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Hero */}
      <View style={styles.hero}>
        <Ionicons name="analytics-outline" size={32} color={colors.primary} />
        <Text style={styles.heroTitle}>Analytics Dashboard</Text>
        <Text style={styles.heroSubtitle}>
          All data is stored locally on your device.
        </Text>
      </View>

      {/* 1. Summary Cards */}
      <View style={styles.summaryRow}>
        <SummaryCard
          icon="pulse-outline"
          iconColor={colors.primary}
          label="Total Screenings"
          value={String(aggregates.totalScreenings)}
        />
        <SummaryCard
          icon="speedometer-outline"
          iconColor={colors.accentDark}
          label="Avg Confidence"
          value={
            aggregates.totalScreenings > 0
              ? `${Math.round(aggregates.avgConfidence * 100)}%`
              : 'N/A'
          }
        />
      </View>
      <View style={styles.summaryRow}>
        <SummaryCard
          icon="trophy-outline"
          iconColor={colors.secondaryDark}
          label="Most Common"
          value={aggregates.mostCommonResult}
        />
        <SummaryCard
          icon="calendar-outline"
          iconColor="#7E57C2"
          label="Last Screening"
          value={lastDateText}
        />
      </View>

      {/* 2. Bar Chart */}
      <SeverityBarChart distribution={aggregates.severityDistribution} />

      {/* 3. Trend Line */}
      <TrendLineChart screeningsPerDay={aggregates.screeningsPerDay} />

      {/* 4. Recent History */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="time-outline" size={20} color={colors.primaryDark} />
          <Text style={styles.cardTitle}>Recent History</Text>
        </View>
        {recentHistory.length === 0 ? (
          <Text style={styles.emptyText}>
            No screening history yet. Complete a screening to see results here.
          </Text>
        ) : (
          recentHistory.map((record) => (
            <View key={record.id} style={styles.historyRow}>
              <View
                style={[
                  styles.historyDot,
                  { backgroundColor: severityColorMap[record.prediction] || colors.primary },
                ]}
              />
              <View style={styles.historyInfo}>
                <Text style={styles.historyPrediction}>{record.prediction}</Text>
                <Text style={styles.historyDate}>
                  {new Date(record.date).toLocaleDateString()} •{' '}
                  {Math.round(record.confidence * 100)}% • {record.mode}
                </Text>
              </View>
            </View>
          ))
        )}
      </Card>

      {/* 5. Usage Insights */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="bulb-outline" size={20} color="#FFB74D" />
          <Text style={styles.cardTitle}>Usage Insights</Text>
        </View>
        {aggregates.totalScreenings === 0 ? (
          <Text style={styles.insightText}>
            Start screening patients to see personalized insights and trends.
          </Text>
        ) : (
          <>
            <Text style={styles.insightText}>
              📊 You have completed {aggregates.totalScreenings} screening
              {aggregates.totalScreenings !== 1 ? 's' : ''} so far.
            </Text>
            {aggregates.avgConfidence > 0 && (
              <Text style={styles.insightText}>
                🎯 Your average prediction confidence is{' '}
                {Math.round(aggregates.avgConfidence * 100)}%.
                {aggregates.avgConfidence >= 0.8
                  ? ' Great image quality!'
                  : ' Try improving lighting for higher confidence.'}
              </Text>
            )}
            {aggregates.mostCommonResult !== 'N/A' && (
              <Text style={styles.insightText}>
                🏆 Most frequent result: {aggregates.mostCommonResult}
              </Text>
            )}
            {Object.keys(aggregates.screeningsPerDay).length > 1 && (
              <Text style={styles.insightText}>
                📅 You've been active for{' '}
                {Object.keys(aggregates.screeningsPerDay).length} day
                {Object.keys(aggregates.screeningsPerDay).length > 1 ? 's' : ''}.
              </Text>
            )}
          </>
        )}
      </Card>
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
  hero: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  heroTitle: {
    ...typography.title,
    color: colors.text,
    marginTop: spacing.sm,
  },
  heroSubtitle: {
    ...typography.caption,
    color: colors.textLight,
    marginTop: spacing.xs,
  },
  // ── Summary Cards ──
  summaryRow: {
    flexDirection: 'row',
    marginBottom: spacing.sm,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginHorizontal: 4,
    alignItems: 'center',
    ...shadows.sm,
  },
  summaryIconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  summaryValue: {
    ...typography.title,
    color: colors.text,
  },
  summaryLabel: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: 2,
  },
  // ── Charts ──
  chartContainer: {
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  emptyChartText: {
    ...typography.body,
    color: colors.textLight,
    textAlign: 'center',
    paddingVertical: spacing.lg,
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
  // ── History ──
  historyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  historyDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.md,
  },
  historyInfo: {
    flex: 1,
  },
  historyPrediction: {
    ...typography.bodyBold,
    color: colors.text,
  },
  historyDate: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  emptyText: {
    ...typography.body,
    color: colors.textLight,
    textAlign: 'center',
    paddingVertical: spacing.md,
  },
  // ── Insights ──
  insightText: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
    lineHeight: 22,
  },
});
