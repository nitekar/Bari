/**
 * colors.ts — Professional baby-tone medical palette
 */

export const colors = {
  // ── Core Palette ──
  primary: '#5B9EC9',         // professional steel blue
  primaryDark: '#3A7FA8',
  primaryLight: '#B8D8ED',    // soft baby blue tint
  secondary: '#E8849A',       // rose
  secondaryDark: '#C96080',
  accent: '#6DC0A0',          // teal-mint
  accentDark: '#4EA080',
  accentLight: '#C2E8DA',

  // ── Surfaces ──
  background: '#F4F8FB',
  surface: '#FFFFFF',
  surfaceElevated: '#EEF4F9',

  // ── Text ──
  text: '#1A2E45',            // deep navy
  textSecondary: '#4A6787',
  textLight: '#8AA5BE',
  textOnPrimary: '#FFFFFF',

  // ── Severity Colors ──
  severityNormal: '#4CAF82',
  severityMild: '#F0B429',
  severityModerate: '#F08C3A',
  severitySevere: '#E05252',

  // ── Severity Backgrounds ──
  severityNormalBg: '#E6F7EF',
  severityMildBg: '#FEF8E6',
  severityModerateBg: '#FEF0E4',
  severitySevereBg: '#FDEAEA',

  // ── UI ──
  border: '#D0E2EE',
  borderLight: '#E8F0F6',
  disabled: '#B0C4D4',
  error: '#E05252',
  errorBg: '#FDEAEA',
  success: '#4CAF82',
  successBg: '#E6F7EF',
  warning: '#F0B429',
  warningBg: '#FEF8E6',
  white: '#FFFFFF',
  overlay: 'rgba(26, 46, 69, 0.45)',
} as const;

/** Map severity label to color */
export const severityColorMap: Record<string, string> = {
  'Non-Anemic': colors.severityNormal,
  'Mild':       colors.severityMild,
  'Moderate':   colors.severityModerate,
  'Severe':     colors.severitySevere,
};

/** Map severity label to background color */
export const severityBgMap: Record<string, string> = {
  'Non-Anemic': colors.severityNormalBg,
  'Mild':       colors.severityMildBg,
  'Moderate':   colors.severityModerateBg,
  'Severe':     colors.severitySevereBg,
};

/** Next screening follow-up interval by severity (days) */
export const nextScreeningDays: Record<string, number> = {
  'Non-Anemic': 180,
  'Mild':        90,
  'Moderate':    30,
  'Severe':      14,
};
