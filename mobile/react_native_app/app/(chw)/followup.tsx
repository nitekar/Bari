/**
 * app/(chw)/followup.tsx — Follow-up placeholder
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing } from '../../src/shared/theme';

export default function FollowupScreen() {
  return (
    <View style={styles.container}>
      <Ionicons name="calendar-outline" size={64} color={colors.textLight} />
      <Text style={styles.title}>No Pending Follow-Ups</Text>
      <Text style={styles.sub}>
        Follow-up reminders will appear here when patients are due for re-screening.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.textSecondary,
    marginTop: spacing.lg,
    textAlign: 'center',
  },
  sub: {
    fontSize: 14,
    color: colors.textLight,
    marginTop: spacing.sm,
    textAlign: 'center',
    lineHeight: 22,
  },
});
