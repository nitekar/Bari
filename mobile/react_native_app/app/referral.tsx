/**
 * app/referral.tsx — Doctor/Referral screen
 *
 * Four sections:
 * 1. Urgency banner (color-coded for moderate/severe)
 * 2. Referral letter generator (auto-populated from screening result)
 * 3. Clinic finder (Rwanda-based health facilities)
 * 4. Emergency contacts (Rwanda health hotline +250 800 22 333)
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Linking,
  Platform,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius, shadows } from '../src/shared/theme';
import { Card, Button } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import { useAnalyticsStore } from '../src/store/analyticsStore';
import { generateAndShareReferralPDF, type ReferralData } from '../src/services/pdfService';

// ── Rwanda Clinic Data ──
const CLINICS = [
  {
    name: 'King Faisal Hospital',
    location: 'Kigali',
    phone: '+250 252 588 888',
    type: 'Referral Hospital',
  },
  {
    name: 'CHUK – University Teaching Hospital',
    location: 'Kigali',
    phone: '+250 252 575 555',
    type: 'Teaching Hospital',
  },
  {
    name: 'Rwanda Military Hospital',
    location: 'Kigali',
    phone: '+250 252 586 422',
    type: 'Military Hospital',
  },
  {
    name: 'Butaro District Hospital',
    location: 'Burera District',
    phone: '+250 252 566 000',
    type: 'District Hospital',
  },
  {
    name: 'Kibagabaga Hospital',
    location: 'Kigali',
    phone: '+250 252 584 471',
    type: 'District Hospital',
  },
];

const EMERGENCY_NUMBER = '+250 800 22 333';

export default function ReferralScreen() {
  const result = useStore((s) => s.result);
  const trackEvent = useAnalyticsStore((s) => s.trackEvent);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

  const isSevere = result?.prediction === 'Severe';
  const isModerate = result?.prediction === 'Moderate';
  const urgencyColor = isSevere ? colors.severitySevere : colors.severityModerate;
  const urgencyBg = isSevere ? colors.severitySevereBg : colors.severityModerateBg;

  const handleCallEmergency = useCallback(() => {
    Linking.openURL(`tel:${EMERGENCY_NUMBER}`).catch(() => {
      if (Platform.OS === 'web') {
        window.alert(`Call Rwanda Health Hotline: ${EMERGENCY_NUMBER}`);
      } else {
        Alert.alert('Unable to make call', `Please dial ${EMERGENCY_NUMBER} manually.`);
      }
    });
  }, []);

  const handleCallClinic = useCallback((phone: string) => {
    Linking.openURL(`tel:${phone}`).catch(() => {
      if (Platform.OS === 'web') {
        window.alert(`Call: ${phone}`);
      } else {
        Alert.alert('Unable to make call', `Please dial ${phone} manually.`);
      }
    });
  }, []);

  const handleGeneratePDF = useCallback(async () => {
    if (!result) return;

    setIsGeneratingPDF(true);
    try {
      const referralData: ReferralData = {
        patientAge: 0, // Will be populated from store if available
        patientGender: 'Female', // Default
        screeningDate: new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        referralAction: result.referral_action,
      };

      await generateAndShareReferralPDF(referralData);
      trackEvent('pdf_exported', { severity: result.prediction });
    } catch (error: any) {
      if (Platform.OS === 'web') {
        window.alert('Failed to generate PDF: ' + (error.message || 'Unknown error'));
      } else {
        Alert.alert('Error', 'Failed to generate the referral letter. Please try again.');
      }
    } finally {
      setIsGeneratingPDF(false);
    }
  }, [result, trackEvent]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* ──────── 1. Urgency Banner ──────── */}
      <View style={[styles.urgencyBanner, { backgroundColor: urgencyBg, borderColor: urgencyColor }]}>
        <View style={[styles.urgencyIconCircle, { backgroundColor: urgencyColor }]}>
          <Ionicons
            name={isSevere ? 'alert-circle' : 'warning'}
            size={28}
            color={colors.white}
          />
        </View>
        <Text style={[styles.urgencyTitle, { color: urgencyColor }]}>
          {isSevere ? '🚨 URGENT: Immediate Attention Required' : '⚠️ Medical Follow-Up Recommended'}
        </Text>
        <Text style={styles.urgencyMessage}>
          {isSevere
            ? 'The screening indicates severe anemia. This condition can be life-threatening. Please seek medical attention immediately or call the emergency hotline below.'
            : 'The screening suggests moderate anemia. We strongly recommend visiting a healthcare provider for a full blood test and clinical evaluation.'}
        </Text>
        {result && (
          <View style={styles.urgencyStats}>
            <View style={styles.urgencyStat}>
              <Text style={styles.urgencyStatLabel}>Result</Text>
              <Text style={[styles.urgencyStatValue, { color: urgencyColor }]}>
                {result.prediction}
              </Text>
            </View>
            <View style={styles.urgencyStatDivider} />
            <View style={styles.urgencyStat}>
              <Text style={styles.urgencyStatLabel}>Confidence</Text>
              <Text style={[styles.urgencyStatValue, { color: urgencyColor }]}>
                {Math.round(result.confidence * 100)}%
              </Text>
            </View>
          </View>
        )}
      </View>

      {/* ──────── 2. Referral Letter Generator ──────── */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="document-text-outline" size={22} color={colors.primaryDark} />
          <Text style={styles.cardTitle}>Referral Letter</Text>
        </View>
        <Text style={styles.cardDescription}>
          Generate a PDF referral letter auto-populated with your screening results. Share it with your healthcare provider.
        </Text>
        <View style={styles.letterPreview}>
          <View style={styles.letterField}>
            <Text style={styles.letterLabel}>Screening Result:</Text>
            <Text style={[styles.letterValue, { color: urgencyColor }]}>
              {result?.prediction || 'N/A'}
            </Text>
          </View>
          <View style={styles.letterField}>
            <Text style={styles.letterLabel}>Confidence:</Text>
            <Text style={styles.letterValue}>
              {result ? `${Math.round(result.confidence * 100)}%` : 'N/A'}
            </Text>
          </View>
          <View style={styles.letterField}>
            <Text style={styles.letterLabel}>Date:</Text>
            <Text style={styles.letterValue}>
              {new Date().toLocaleDateString()}
            </Text>
          </View>
          <View style={styles.letterField}>
            <Text style={styles.letterLabel}>Recommendation:</Text>
            <Text style={styles.letterValue} numberOfLines={2}>
              {result?.referral_action || 'N/A'}
            </Text>
          </View>
        </View>
        <Button
          title="Generate & Share PDF"
          onPress={handleGeneratePDF}
          variant="primary"
          loading={isGeneratingPDF}
          icon={
            !isGeneratingPDF ? (
              <Ionicons name="download-outline" size={18} color={colors.white} />
            ) : undefined
          }
        />
      </Card>

      {/* ──────── 3. Clinic Finder ──────── */}
      <Card>
        <View style={styles.cardHeader}>
          <Ionicons name="location-outline" size={22} color={colors.accentDark} />
          <Text style={styles.cardTitle}>Nearby Health Facilities</Text>
        </View>
        <Text style={styles.cardDescription}>
          Recommended health facilities in Rwanda for anemia evaluation and treatment.
        </Text>
        {CLINICS.map((clinic, i) => (
          <TouchableOpacity
            key={i}
            style={styles.clinicRow}
            onPress={() => handleCallClinic(clinic.phone)}
            activeOpacity={0.7}
          >
            <View style={styles.clinicIcon}>
              <Ionicons name="medical" size={18} color={colors.primary} />
            </View>
            <View style={styles.clinicInfo}>
              <Text style={styles.clinicName}>{clinic.name}</Text>
              <Text style={styles.clinicDetail}>
                {clinic.location} • {clinic.type}
              </Text>
              <Text style={[styles.clinicPhone, { color: colors.primary }]}>
                {clinic.phone}
              </Text>
            </View>
            <Ionicons name="call-outline" size={20} color={colors.primary} />
          </TouchableOpacity>
        ))}
      </Card>

      {/* ──────── 4. Emergency Contacts ──────── */}
      <Card style={styles.emergencyCard}>
        <View style={styles.emergencyHeader}>
          <View style={styles.emergencyIconCircle}>
            <Ionicons name="call" size={24} color={colors.white} />
          </View>
          <View style={styles.emergencyTextContainer}>
            <Text style={styles.emergencyTitle}>Rwanda Health Emergency</Text>
            <Text style={styles.emergencySubtitle}>Available 24/7</Text>
          </View>
        </View>
        <TouchableOpacity
          style={styles.emergencyPhoneButton}
          onPress={handleCallEmergency}
          activeOpacity={0.7}
        >
          <Ionicons name="call" size={20} color={colors.white} />
          <Text style={styles.emergencyPhone}>{EMERGENCY_NUMBER}</Text>
        </TouchableOpacity>
        <Text style={styles.emergencyDisclaimer}>
          Call this number if you or someone is experiencing symptoms of severe anemia including extreme fatigue, rapid heartbeat, or difficulty breathing.
        </Text>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing.xxl,
  },
  // ── Urgency Banner ──
  urgencyBanner: {
    borderWidth: 2,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    alignItems: 'center',
  },
  urgencyIconCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  urgencyTitle: {
    ...typography.subtitle,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  urgencyMessage: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  urgencyStats: {
    flexDirection: 'row',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  urgencyStat: {
    flex: 1,
    alignItems: 'center',
  },
  urgencyStatDivider: {
    width: 1,
    backgroundColor: colors.border,
  },
  urgencyStatLabel: {
    ...typography.caption,
    color: colors.textLight,
  },
  urgencyStatValue: {
    ...typography.title,
    marginTop: 2,
  },
  // ── Cards ──
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  cardTitle: {
    ...typography.subtitle,
    color: colors.text,
    marginLeft: spacing.sm,
  },
  cardDescription: {
    ...typography.body,
    color: colors.textSecondary,
    marginBottom: spacing.md,
    lineHeight: 22,
  },
  // ── Letter Preview ──
  letterPreview: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  letterField: {
    flexDirection: 'row',
    marginBottom: spacing.sm,
  },
  letterLabel: {
    ...typography.captionBold,
    color: colors.textSecondary,
    width: 130,
  },
  letterValue: {
    ...typography.body,
    color: colors.text,
    flex: 1,
  },
  // ── Clinics ──
  clinicRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  clinicIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  clinicInfo: {
    flex: 1,
  },
  clinicName: {
    ...typography.bodyBold,
    color: colors.text,
  },
  clinicDetail: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 2,
  },
  clinicPhone: {
    ...typography.captionBold,
    marginTop: 2,
  },
  // ── Emergency ──
  emergencyCard: {
    backgroundColor: '#FFEBEE',
    borderWidth: 2,
    borderColor: colors.error,
  },
  emergencyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  emergencyIconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.error,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  emergencyTextContainer: {
    flex: 1,
  },
  emergencyTitle: {
    ...typography.subtitle,
    color: colors.error,
  },
  emergencySubtitle: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  emergencyPhoneButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.error,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.md,
    marginBottom: spacing.md,
  },
  emergencyPhone: {
    ...typography.title,
    color: colors.white,
    marginLeft: spacing.sm,
  },
  emergencyDisclaimer: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
