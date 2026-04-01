/**
 * LanguageSwitch.tsx — Global language toggle for headerRight
 * Toggles between en, fr, rw
 */
import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation, type Language } from '../../i18n';
import { colors, spacing, borderRadius } from '../theme';

const LANG_CYCLE: Language[] = ['en', 'fr', 'rw'];
const LANG_LABELS: Record<Language, string> = {
  en: 'EN',
  fr: 'FR',
  rw: 'RW',
};

export default function LanguageSwitch() {
  const { lang, setLang } = useTranslation();

  const handleToggle = () => {
    const currentIndex = LANG_CYCLE.indexOf(lang);
    const nextIndex = (currentIndex + 1) % LANG_CYCLE.length;
    setLang(LANG_CYCLE[nextIndex]);
  };

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={handleToggle}
      activeOpacity={0.7}
      accessibilityRole="button"
      accessibilityLabel={`Current language: ${LANG_LABELS[lang]}. Switch language.`}
    >
      <Ionicons name="globe-outline" size={18} color={colors.primaryDark} />
      <Text style={styles.text}>{LANG_LABELS[lang]}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44, // Accessibility requirement
    minWidth: 44,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.md,
    marginRight: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    gap: 6,
  },
  text: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primaryDark,
  },
});
