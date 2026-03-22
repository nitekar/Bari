/**
 * OfflineIndicator.tsx — Banner shown when the device is offline
 *
 * Displays stale data timestamp and pending queue count.
 */
import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../theme';

interface OfflineIndicatorProps {
  isConnected: boolean;
  queueCount: number;
  lastSyncedAt?: Date | null;
}

const OfflineIndicator: React.FC<OfflineIndicatorProps> = ({
  isConnected,
  queueCount,
  lastSyncedAt,
}) => {
  if (isConnected) return null;

  const staleText = lastSyncedAt
    ? `Last synced: ${lastSyncedAt.toLocaleTimeString()}`
    : 'Never synced';

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <Ionicons name="cloud-offline-outline" size={18} color={colors.white} />
        <Text style={styles.text}>You are offline</Text>
        {queueCount > 0 && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>{queueCount}</Text>
          </View>
        )}
      </View>
      <Text style={styles.staleText}>{staleText}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#616161',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  text: {
    ...typography.captionBold,
    color: colors.white,
    marginLeft: spacing.sm,
  },
  badge: {
    backgroundColor: colors.error,
    borderRadius: borderRadius.full,
    minWidth: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: spacing.sm,
    paddingHorizontal: 6,
  },
  badgeText: {
    ...typography.caption,
    color: colors.white,
    fontSize: 11,
    fontWeight: '700',
  },
  staleText: {
    ...typography.caption,
    color: '#BDBDBD',
    marginTop: 2,
  },
});

export default memo(OfflineIndicator);
