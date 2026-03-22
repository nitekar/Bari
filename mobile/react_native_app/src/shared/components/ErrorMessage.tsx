/**
 * ErrorMessage.tsx — Error card with retry button
 */
import React, { memo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../theme';

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onRetry }) => {
  return (
    <View style={styles.container}>
      <Ionicons name="alert-circle" size={24} color={colors.error} />
      <Text style={styles.text}>{message}</Text>
      {onRetry && (
        <TouchableOpacity style={styles.retryButton} onPress={onRetry}>
          <Ionicons name="refresh" size={18} color={colors.primary} />
          <Text style={styles.retryText}>Try Again</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.errorBg,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  text: {
    ...typography.body,
    color: colors.error,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  retryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.sm,
  },
  retryText: {
    ...typography.captionBold,
    color: colors.primary,
    marginLeft: spacing.xs,
  },
});

export default memo(ErrorMessage);
