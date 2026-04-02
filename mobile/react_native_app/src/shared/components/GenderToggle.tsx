/**
 * GenderToggle.tsx — Male / Female toggle selector
 */
import React, { memo } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius, shadows } from '../theme';

interface GenderToggleProps {
  value: number; // 0 = Male, 1 = Female
  onChange: (gender: number) => void;
}

const GenderToggle: React.FC<GenderToggleProps> = ({ value, onChange }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>GENDER</Text>
      <View style={styles.toggleRow}>
        <TouchableOpacity
          style={[
            styles.option,
            value === 0 && styles.optionActive,
            value === 0 && { borderColor: colors.primary },
          ]}
          onPress={() => onChange(0)}
          activeOpacity={0.7}
        >
          <Ionicons
            name="male"
            size={24}
            color={value === 0 ? colors.primary : colors.textLight}
          />
          <Text
            style={[
              styles.optionText,
              value === 0 && { color: colors.primary, fontWeight: '700' },
            ]}
          >
            Male
          </Text>
        </TouchableOpacity>

        <View style={{ width: spacing.md }} />

        <TouchableOpacity
          style={[
            styles.option,
            value === 1 && styles.optionActive,
            value === 1 && { borderColor: colors.secondary },
          ]}
          onPress={() => onChange(1)}
          activeOpacity={0.7}
        >
          <Ionicons
            name="female"
            size={24}
            color={value === 1 ? colors.secondaryDark : colors.textLight}
          />
          <Text
            style={[
              styles.optionText,
              value === 1 && { color: colors.secondaryDark, fontWeight: '700' },
            ]}
          >
            Female
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  label: {
    ...typography.captionBold,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.5,
  },
  toggleRow: {
    flexDirection: 'row',
  },
  option: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.md,
    borderWidth: 2,
    borderColor: colors.border,
    ...shadows.sm,
  },
  optionActive: {
    backgroundColor: colors.surface,
  },
  optionText: {
    ...typography.bodyBold,
    color: colors.textLight,
    marginLeft: spacing.sm,
  },
});

export default memo(GenderToggle);
