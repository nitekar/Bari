/**
 * ResultBadge.tsx — Color-coded severity badge
 */
import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { typography, spacing, borderRadius } from '../theme';
import { severityColorMap, severityBgMap, colors } from '../theme/colors';

interface ResultBadgeProps {
  severity: string;
  confidence: number;
}

const severityIconMap: Record<string, keyof typeof Ionicons.glyphMap> = {
  'Non-Anemic': 'checkmark-circle',
  'Mild': 'alert-circle',
  'Moderate': 'warning',
  'Severe': 'close-circle',
};

const ResultBadge: React.FC<ResultBadgeProps> = ({ severity, confidence }) => {
  const badgeColor = severityColorMap[severity] || colors.primary;
  const bgColor = severityBgMap[severity] || colors.surfaceElevated;
  const icon = severityIconMap[severity] || 'help-circle';

  return (
    <View style={[styles.container, { backgroundColor: bgColor, borderColor: badgeColor }]}>
      <View style={[styles.iconCircle, { backgroundColor: badgeColor }]}>
        <Ionicons name={icon} size={28} color={colors.white} />
      </View>
      <Text style={[styles.label, { color: badgeColor }]}>{severity}</Text>
      <Text style={styles.confidence}>
        {Math.round(confidence * 100)}% confidence
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.xl,
    borderWidth: 2,
    marginBottom: spacing.md,
  },
  iconCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  label: {
    ...typography.badge,
    fontSize: 20,
    marginBottom: spacing.xs,
  },
  confidence: {
    ...typography.body,
    color: colors.textSecondary,
  },
});

export default memo(ResultBadge);
