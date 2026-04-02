/**
 * Card.tsx — Rounded card with soft shadow
 */
import React, { memo } from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { colors, spacing, borderRadius, shadows } from '../theme';

interface CardProps {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  variant?: 'default' | 'elevated';
}

const Card: React.FC<CardProps> = ({
  children,
  style,
  variant = 'default',
}) => {
  return (
    <View
      style={[
        styles.card,
        variant === 'elevated' ? shadows.lg : shadows.md,
        style,
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
  },
});

export default memo(Card);
