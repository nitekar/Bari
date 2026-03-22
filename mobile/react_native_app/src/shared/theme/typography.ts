/**
 * typography.ts — Font style presets
 */
import { TextStyle } from 'react-native';

export const typography: Record<string, TextStyle> = {
  heroTitle: {
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: 17,
    fontWeight: '600',
  },
  body: {
    fontSize: 15,
    fontWeight: '400',
    lineHeight: 22,
  },
  bodyBold: {
    fontSize: 15,
    fontWeight: '600',
    lineHeight: 22,
  },
  caption: {
    fontSize: 13,
    fontWeight: '400',
    lineHeight: 18,
  },
  captionBold: {
    fontSize: 13,
    fontWeight: '600',
    lineHeight: 18,
  },
  button: {
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  badge: {
    fontSize: 14,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
} as const;
