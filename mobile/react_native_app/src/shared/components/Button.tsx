/**
 * Button.tsx — Reusable rounded button with variants
 */
import React, { memo } from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { colors, typography, spacing, borderRadius } from '../theme';

type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'danger';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: ButtonVariant;
  loading?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
  icon?: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  loading = false,
  disabled = false,
  style,
  icon,
}) => {
  const isDisabled = disabled || loading;

  const variantStyles = getVariantStyles(variant, isDisabled);

  return (
    <TouchableOpacity
      style={[styles.base, variantStyles.container, style]}
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator color={variantStyles.textColor} size="small" />
      ) : (
        <>
          {icon && <>{icon}</>}
          <Text style={[styles.text, { color: variantStyles.textColor }, icon ? { marginLeft: spacing.sm } : undefined]}>
            {title}
          </Text>
        </>
      )}
    </TouchableOpacity>
  );
};

function getVariantStyles(variant: ButtonVariant, disabled: boolean) {
  const base = {
    container: {} as ViewStyle,
    textColor: colors.white,
  };

  if (disabled) {
    return {
      container: { backgroundColor: colors.disabled },
      textColor: colors.white,
    };
  }

  switch (variant) {
    case 'primary':
      return {
        container: { backgroundColor: colors.primary },
        textColor: colors.white,
      };
    case 'secondary':
      return {
        container: { backgroundColor: colors.secondary },
        textColor: colors.text,
      };
    case 'outline':
      return {
        container: {
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderColor: colors.primary,
        },
        textColor: colors.primary,
      };
    case 'danger':
      return {
        container: { backgroundColor: colors.error },
        textColor: colors.white,
      };
    default:
      return base;
  }
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.lg,
    minHeight: 52,
  },
  text: {
    ...typography.button,
  },
});

export default memo(Button);
