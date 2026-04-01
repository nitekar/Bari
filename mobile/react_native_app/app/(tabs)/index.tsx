/**
 * app/(tabs)/index.tsx — Dashboard
 */
import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { severityColorMap, nextScreeningDays } from '../../src/shared/theme/colors';
import { Button, Logo } from '../../src/shared/components';
import { useStore, type ScreeningRecord } from '../../src/store/useStore';
import { useTranslation } from '../../src/i18n';

export default function HomeScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const reset = useStore((s) => s.reset);
  const history = useStore((s) => s.history);
  const recent = history.slice(0, 3);

  const handleStartScreening = () => {
    reset();
    router.push('/screening');
  };

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.container}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Header ── */}
      <View style={styles.header}>
        <Logo size="md" horizontal showText />
        <TouchableOpacity onPress={() => router.push('/settings')} style={styles.settingsBtn}>
          <Ionicons name="settings-outline" size={22} color={colors.textSecondary} />
        </TouchableOpacity>
      </View>

      {/* ── Hero banner ── */}
      <View style={styles.hero}>
        <View style={styles.bubble1} />
        <View style={styles.bubble2} />
        <Text style={styles.heroTitle}>{t.home.subtitle}</Text>
        <Button
          title={t.home.startScreening}
          onPress={handleStartScreening}
          variant="primary"
          icon={<Ionicons name="medkit-outline" size={20} color={colors.white} />}
        />
      </View>

      {/* ── Quick actions ── */}
      <View style={styles.cardRow}>
        <InfoCard
          icon="camera-outline"
          color={colors.severityNormal}
          bg={colors.severityNormalBg}
          title={t.home.imageAnalysis}
          sub={t.home.conjunctivaPhoto}
          onPress={handleStartScreening}
        />
        <InfoCard
          icon="analytics-outline"
          color={colors.severityModerate}
          bg={colors.severityMildBg}
          title={t.home.aiPrediction}
          sub={t.home.severityConfidence}
          onPress={handleStartScreening}
        />
        <InfoCard
          icon="book-outline"
          color="#9575CD"
          bg="#EDE7F6"
          title={t.home.learnMore}
          sub={t.home.nutritionAdvice}
          onPress={() => router.push('/education')}
        />
      </View>

      {/* ── Recent patients ── */}
      {recent.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>{t.guestDashboard.recentPatients}</Text>
            <TouchableOpacity onPress={() => router.push('/history')}>
              <Text style={styles.seeAll}>{t.guestDashboard.seeAll}</Text>
            </TouchableOpacity>
          </View>
          {recent.map((record) => (
            <PatientCard key={record.id} record={record} />
          ))}
        </View>
      )}

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function PatientCard({ record }: { record: ScreeningRecord }) {
  const { t } = useTranslation();
  const severityColor = severityColorMap[record.prediction] ?? colors.primary;
  const days = nextScreeningDays[record.prediction] ?? 90;
  const nextDate = new Date(record.date);
  nextDate.setDate(nextDate.getDate() + days);
  const nextDateStr = nextDate.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  const screeningDate = new Date(record.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  const isPast = nextDate < new Date();

  return (
    <View style={styles.patientCard}>
      <View style={[styles.accentBar, { backgroundColor: severityColor }]} />
      <View style={styles.patientBody}>
        {/* Name + severity */}
        <View style={styles.patientTopRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.patientName} numberOfLines={1}>
              {record.patientName || t.guestDashboard.unknownPatient}
            </Text>
            <View style={styles.metaRow}>
              {record.age != null && (
                <View style={styles.metaChip}>
                  <Ionicons name="person-outline" size={11} color={colors.textSecondary} />
                  <Text style={styles.metaText}>{record.age} {t.guestDashboard.yrs}</Text>
                </View>
              )}
              {record.patientLocation ? (
                <View style={styles.metaChip}>
                  <Ionicons name="location-outline" size={11} color={colors.textSecondary} />
                  <Text style={styles.metaText}>{record.patientLocation}</Text>
                </View>
              ) : null}
              <View style={styles.metaChip}>
                <Ionicons name="calendar-outline" size={11} color={colors.textSecondary} />
                <Text style={styles.metaText}>{screeningDate}</Text>
              </View>
            </View>
          </View>
          <View style={[styles.severityPill, { backgroundColor: severityColor + '20', borderColor: severityColor }]}>
            <Text style={[styles.severityPillText, { color: severityColor }]}>{record.prediction}</Text>
          </View>
        </View>

        {/* Next screening reminder */}
        <View style={[styles.reminderRow, { backgroundColor: isPast ? colors.severitySevereBg : colors.accentLight }]}>
          <Ionicons
            name="alarm-outline"
            size={13}
            color={isPast ? colors.severitySevere : colors.accentDark}
          />
          <Text style={[styles.reminderText, { color: isPast ? colors.severitySevere : colors.accentDark }]}>
            {isPast ? t.guestDashboard.followUpOverdue : t.guestDashboard.nextScreening}
            <Text style={{ fontWeight: '700' }}>{nextDateStr}</Text>
          </Text>
        </View>
      </View>
    </View>
  );
}

function InfoCard({
  icon, color, bg, title, sub, onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  bg: string;
  title: string;
  sub: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={[styles.infoCard, { backgroundColor: bg }]} onPress={onPress} activeOpacity={0.8}>
      <View style={[styles.infoIconCircle, { backgroundColor: color + '30' }]}>
        <Ionicons name={icon} size={22} color={color} />
      </View>
      <Text style={styles.infoTitle}>{title}</Text>
      <Text style={styles.infoSub}>{sub}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  container: { paddingHorizontal: spacing.lg, paddingTop: 52 },
  // ── Header ──
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
  },
  settingsBtn: { padding: 6 },
  // ── Hero ──
  hero: {
    backgroundColor: colors.primary,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.lg,
    overflow: 'hidden',
    position: 'relative',
  },
  bubble1: {
    position: 'absolute', top: -20, right: -20,
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  bubble2: {
    position: 'absolute', bottom: -30, left: -10,
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  heroTitle: {
    fontSize: 15, color: 'rgba(255,255,255,0.9)',
    lineHeight: 22, marginBottom: spacing.lg,
  },
  // ── Info cards ──
  cardRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
    gap: 8,
  },
  infoCard: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: 6,
    borderRadius: borderRadius.lg,
    ...shadows.sm,
  },
  infoIconCircle: {
    width: 44, height: 44, borderRadius: 22,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  infoTitle: { fontSize: 12, fontWeight: '700', color: colors.text, textAlign: 'center' },
  infoSub: { fontSize: 10, color: colors.textSecondary, textAlign: 'center', marginTop: 2 },
  // ── Section ──
  section: { marginBottom: spacing.lg },
  sectionHeader: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between', marginBottom: spacing.md,
  },
  sectionTitle: { fontSize: 16, fontWeight: '700', color: colors.text },
  seeAll: { fontSize: 13, color: colors.primary, fontWeight: '600' },
  // ── Patient card ──
  patientCard: {
    flexDirection: 'row',
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    marginBottom: spacing.sm,
    ...shadows.sm,
  },
  accentBar: { width: 5 },
  patientBody: { flex: 1, padding: spacing.md },
  patientTopRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 8 },
  patientName: { fontSize: 15, fontWeight: '700', color: colors.text },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 4 },
  metaChip: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  metaText: { fontSize: 11, color: colors.textSecondary },
  severityPill: {
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: 20, borderWidth: 1,
    marginLeft: 8, alignSelf: 'flex-start',
  },
  severityPillText: { fontSize: 11, fontWeight: '700' },
  reminderRow: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 8, marginTop: 4,
  },
  reminderText: { fontSize: 12, flex: 1 },
});
