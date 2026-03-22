/**
 * colors.ts — Baby-friendly medical color palette
 */

export const colors = {
  // ── Core Palette ──
  primary: '#A7C7E7',     // soft blue
  primaryDark: '#7BA7D0',
  secondary: '#F6C1CC',   // soft pink
  secondaryDark: '#E8A0AF',
  accent: '#B5EAD7',      // mint green
  accentDark: '#8DD4BA',

  // ── Surfaces ──
  background: '#FAFAFA',
  surface: '#FFFFFF',
  surfaceElevated: '#F5F5F5',

  // ── Text ──
  text: '#333333',
  textSecondary: '#666666',
  textLight: '#999999',
  textOnPrimary: '#FFFFFF',

  // ── Severity Colors ──
  severityNormal: '#81C784',     // green
  severityMild: '#FFD54F',       // yellow
  severityModerate: '#FFB74D',   // orange
  severitySevere: '#E57373',     // red

  // ── Severity Backgrounds (lighter) ──
  severityNormalBg: '#E8F5E9',
  severityMildBg: '#FFF8E1',
  severityModerateBg: '#FFF3E0',
  severitySevereBg: '#FFEBEE',

  // ── UI ──
  border: '#E0E0E0',
  disabled: '#BDBDBD',
  error: '#E57373',
  errorBg: '#FFEBEE',
  white: '#FFFFFF',
  overlay: 'rgba(0, 0, 0, 0.4)',
} as const;

/** Map severity label to color */
export const severityColorMap: Record<string, string> = {
  'Non-Anemic': colors.severityNormal,
  'Mild': colors.severityMild,
  'Moderate': colors.severityModerate,
  'Severe': colors.severitySevere,
};

/** Map severity label to background color */
export const severityBgMap: Record<string, string> = {
  'Non-Anemic': colors.severityNormalBg,
  'Mild': colors.severityMildBg,
  'Moderate': colors.severityModerateBg,
  'Severe': colors.severitySevereBg,
};
