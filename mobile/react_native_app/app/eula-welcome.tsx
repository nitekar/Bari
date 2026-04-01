import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert, SafeAreaView } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography, shadows } from '../src/shared/theme';
import { Button, Logo } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

export default function EulaWelcomeScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const setHasAcceptedEula = useStore((s) => s.setHasAcceptedEula);
  const setUserId = useStore((s) => s.setUserId);
  const [scrolledToBottom, setScrolledToBottom] = useState(false);

  const handleScroll = (event: any) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    const isCloseToBottom = layoutMeasurement.height + contentOffset.y >= contentSize.height - 20;
    if (isCloseToBottom && !scrolledToBottom) setScrolledToBottom(true);
  };

  const handleAccept = () => {
    setHasAcceptedEula(true);
    // The layout interceptor will redirect immediately based on role state!
    router.replace('/'); 
  };

  const handleDecline = () => {
    Alert.alert(
      t.eula.consentRequired,
      t.eula.consentMsg,
      [
        { text: t.eula.cancel, style: 'cancel' },
        { 
          text: t.eula.exitToSignIn, 
          style: 'destructive',
          onPress: () => {
            setUserId(null); // Clear session if any
            router.replace('/auth');
          }
        }
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <Logo size="md" horizontal={false} showText={false} />
          <Text style={styles.title}>{t.eula.title}</Text>
          <Text style={styles.subtitle}>{t.eula.subtitle}</Text>
        </View>

        {/* Scrollable Terms */}
        <ScrollView 
          style={styles.scroll} 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          onScroll={handleScroll}
          scrollEventThrottle={400}
        >
          {/* Sections mapping without global SECTIONS const */}
          {[
            {
              icon: 'shield-checkmark-outline' as const,
              title: t.eula.section1Title,
              body: t.eula.section1Body,
            },
            {
              icon: 'medical-outline' as const,
              title: t.eula.section2Title,
              body: t.eula.section2Body,
            },
            {
              icon: 'document-text-outline' as const,
              title: t.eula.section3Title,
              body: t.eula.section3Body,
            },
          ].map((sec, idx) => (
            <View key={idx} style={styles.card}>
              <View style={styles.cardHeader}>
                <View style={styles.iconBox}>
                  <Ionicons name={sec.icon} size={20} color={colors.primary} />
                </View>
                <Text style={styles.cardTitle}>{sec.title}</Text>
              </View>
              <Text style={styles.cardBody}>{sec.body}</Text>
            </View>
          ))}
          <View style={{ height: 20 }} />
          <Text style={styles.readMoreAlert}>
            {t.eula.endOfDoc}
          </Text>
        </ScrollView>

        {/* Action Buttons */}
        <View style={styles.footer}>
          <Button
            title={t.eula.acceptBtn}
            onPress={handleAccept}
            variant="primary"
            disabled={!scrolledToBottom}
            icon={<Ionicons name="checkmark-circle-outline" size={20} color={scrolledToBottom ? colors.white : colors.textLight} />}
          />
          <TouchableOpacity onPress={handleDecline} style={styles.declineBtn}>
            <Text style={styles.declineText}>{t.eula.declineBtn}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  container: { flex: 1 },
  header: {
    padding: spacing.xl,
    alignItems: 'center',
    paddingBottom: spacing.lg,
  },
  title: {
    ...typography.h2,
    color: colors.text,
    marginTop: spacing.md,
  },
  subtitle: {
    ...typography.body,
    color: colors.textSecondary,
    marginTop: 4,
  },
  scroll: {
    flex: 1,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FAF9F6',
  },
  scrollContent: {
    padding: spacing.lg,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    ...shadows.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
    gap: spacing.sm,
  },
  iconBox: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTitle: {
    ...typography.bodyBold,
    color: colors.text,
    fontSize: 15,
  },
  cardBody: {
    ...typography.body,
    color: colors.textSecondary,
    lineHeight: 22,
    fontSize: 13,
  },
  readMoreAlert: {
    textAlign: 'center',
    color: colors.textLight,
    fontSize: 12,
    fontStyle: 'italic',
    marginBottom: spacing.xxl,
  },
  footer: {
    padding: spacing.lg,
    paddingTop: spacing.md,
    backgroundColor: colors.surface,
    ...shadows.md,
  },
  declineBtn: {
    marginTop: spacing.md,
    alignItems: 'center',
  },
  declineText: {
    fontSize: 14,
    color: colors.textLight,
    fontWeight: '600',
  },
});
