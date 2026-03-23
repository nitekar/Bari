/**
 * Logo.tsx — Baby-themed app logo
 * Eye icon (conjunctiva screening) + heart badge (care)
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../theme/colors';

type LogoSize = 'sm' | 'md' | 'lg';

interface LogoProps {
  size?: LogoSize;
  showText?: boolean;
  horizontal?: boolean;
}

const SIZES: Record<LogoSize, { circle: number; icon: number; heart: number; title: number; sub: number }> = {
  sm: { circle: 36, icon: 18, heart: 10, title: 15, sub: 11 },
  md: { circle: 52, icon: 26, heart: 14, title: 20, sub: 13 },
  lg: { circle: 72, icon: 36, heart: 18, title: 26, sub: 16 },
};

export default function Logo({ size = 'md', showText = true, horizontal = true }: LogoProps) {
  const s = SIZES[size];

  const mark = (
    <View
      style={[
        styles.circle,
        { width: s.circle, height: s.circle, borderRadius: s.circle / 2 },
      ]}
    >
      <Ionicons name="eye" size={s.icon} color={colors.white} />
      {/* Heart badge — bottom-right */}
      <View
        style={[
          styles.heartBadge,
          { width: s.heart + 4, height: s.heart + 4, borderRadius: (s.heart + 4) / 2 },
        ]}
      >
        <Ionicons name="heart" size={s.heart} color={colors.secondary} />
      </View>
    </View>
  );

  const label = showText ? (
    <View style={horizontal ? styles.labelH : styles.labelV}>
      <Text style={[styles.bold, { fontSize: s.title }]}>Bari</Text>
      <Text style={[styles.light, { fontSize: s.sub }]}>Anemia</Text>
    </View>
  ) : null;

  return (
    <View style={horizontal ? styles.rowContainer : styles.colContainer}>
      {mark}
      {label}
    </View>
  );
}

const styles = StyleSheet.create({
  rowContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  colContainer: {
    alignItems: 'center',
    gap: 8,
  },
  circle: {
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heartBadge: {
    position: 'absolute',
    bottom: -2,
    right: -2,
    backgroundColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 2,
  },
  labelH: {
    flexDirection: 'column',
  },
  labelV: {
    alignItems: 'center',
  },
  bold: {
    fontWeight: '800',
    color: colors.primaryDark,
    lineHeight: 22,
  },
  light: {
    fontWeight: '400',
    color: colors.textSecondary,
    lineHeight: 16,
  },
});
