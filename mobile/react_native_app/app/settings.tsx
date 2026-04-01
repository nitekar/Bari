/**
 * app/settings.tsx — Language and app preferences
 */
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { Card, Logo, InputField } from '../src/shared/components';
import { useTranslation, type Language } from '../src/i18n';
import { useStore } from '../src/store/useStore';

const LANGUAGES: { code: Language; label: string; flag: string }[] = [
  { code: 'en', label: 'English', flag: '🇬🇧' },
  { code: 'fr', label: 'Français', flag: '🇫🇷' },
  { code: 'rw', label: 'Kinyarwanda', flag: '🇷🇼' },
];

export default function SettingsScreen() {
  const { t, lang, setLang } = useTranslation();
  const userName = useStore((s) => s.userName);
  const setUserName = useStore((s) => s.setUserName);

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Profile Setup ── */}
      <Text style={styles.sectionTitle}>Account Setup</Text>
      <Card style={{ marginBottom: spacing.lg }}>
        <InputField
          label="Your Full Name"
          value={userName || ''}
          onChangeText={setUserName}
          placeholder="e.g. Marie Claire"
        />
      </Card>

      {/* ── Language ── */}
      <Text style={styles.sectionTitle}>{t.settings.language}</Text>
      <Text style={styles.sectionSub}>{t.settings.selectLang}</Text>

      <Card style={styles.langCard}>
        {LANGUAGES.map((l, i) => {
          const active = lang === l.code;
          return (
            <React.Fragment key={l.code}>
              {i > 0 && <View style={styles.divider} />}
              <TouchableOpacity
                style={[styles.langRow, active && styles.langRowActive]}
                onPress={() => setLang(l.code)}
                activeOpacity={0.7}
              >
                <Text style={styles.flag}>{l.flag}</Text>
                <Text style={[styles.langLabel, active && styles.langLabelActive]}>
                  {l.label}
                </Text>
                {active && (
                  <Ionicons name="checkmark-circle" size={22} color={colors.primary} />
                )}
              </TouchableOpacity>
            </React.Fragment>
          );
        })}
      </Card>

      {/* ── About ── */}
      <Text style={styles.sectionTitle}>{t.settings.about}</Text>
      <Card style={styles.aboutCard}>
        <View style={styles.aboutLogo}>
          <Logo size="md" horizontal />
        </View>
        <View style={styles.divider} />
        <InfoRow label={t.settings.version} value="2.0.0" />
        <View style={styles.divider} />
        <View style={styles.descRow}>
          <Text style={styles.descText}>{t.settings.description}</Text>
        </View>
      </Card>
    </ScrollView>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },
  // ── Section ──
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 4,
    marginTop: spacing.md,
  },
  sectionSub: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: spacing.md,
  },
  // ── Language card ──
  langCard: { paddingVertical: 0, overflow: 'hidden', marginBottom: spacing.lg },
  langRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    gap: spacing.md,
  },
  langRowActive: {
    backgroundColor: colors.primary + '12',
  },
  flag: { fontSize: 22 },
  langLabel: {
    flex: 1,
    fontSize: 16,
    fontWeight: '500',
    color: colors.text,
  },
  langLabelActive: {
    fontWeight: '700',
    color: colors.primaryDark,
  },
  divider: { height: 1, backgroundColor: colors.border },
  // ── About card ──
  aboutCard: { paddingVertical: 0, overflow: 'hidden' },
  aboutLogo: {
    padding: spacing.lg,
    alignItems: 'center',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  infoLabel: { fontSize: 14, color: colors.textSecondary },
  infoValue: { fontSize: 14, fontWeight: '600', color: colors.text },
  descRow: { padding: spacing.md },
  descText: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 20,
    fontStyle: 'italic',
  },
});
