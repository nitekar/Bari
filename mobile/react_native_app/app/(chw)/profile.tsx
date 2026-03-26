/**
 * app/(chw)/profile.tsx — CHW Profile, settings link, sign-out
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card, Button, Logo } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { useAnalyticsStore } from '../../src/store/analyticsStore';
import { signOut } from '../../src/services/supabaseAuth';
import { isSupabaseConfigured } from '../../src/services/supabase';
import { useTranslation } from '../../src/i18n';

export default function ChwProfileScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const userId = useStore((s) => s.userId);
  const history = useStore((s) => s.history);
  const setUserId = useStore((s) => s.setUserId);
  const clearHistory = useStore((s) => s.clearHistory);
  const clearEvents = useAnalyticsStore((s) => s.clearEvents);
  const [signingOut, setSigningOut] = useState(false);

  const handleSignOut = useCallback(async () => {
    const doSignOut = async () => {
      setSigningOut(true);
      try {
        await signOut();
        setUserId(null);
      } catch {
        // ignore
      } finally {
        setSigningOut(false);
      }
    };

    if (Platform.OS === 'web') {
      if (window.confirm('Sign out?')) doSignOut();
    } else {
      Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign Out', style: 'destructive', onPress: doSignOut },
      ]);
    }
  }, [setUserId]);

  const initials = userId ? userId.slice(0, 2).toUpperCase() : '??';

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Card style={styles.profileCard}>
        <View style={styles.avatarCircle}>
          {userId ? (
            <Text style={styles.avatarText}>{initials}</Text>
          ) : (
            <Ionicons name="person" size={32} color={colors.white} />
          )}
        </View>
        {userId ? (
          <>
            <Text style={styles.roleLabel}>Community Health Worker</Text>
            <Text style={styles.userId} numberOfLines={1}>{userId}</Text>
          </>
        ) : (
          <>
            <Text style={styles.guestLabel}>{t.profile.guestMode}</Text>
            <Text style={styles.notSignedIn}>{t.profile.notSignedIn}</Text>
          </>
        )}
        <View style={styles.statBadge}>
          <Text style={styles.statNumber}>{history.length}</Text>
          <Text style={styles.statLabel}>{t.profile.totalScreenings}</Text>
        </View>
      </Card>

      {isSupabaseConfigured && (
        <View style={styles.section}>
          {userId ? (
            <Button
              title={signingOut ? t.profile.signingOut : t.profile.signOut}
              onPress={handleSignOut}
              variant="outline"
              loading={signingOut}
              icon={<Ionicons name="log-out-outline" size={20} color={colors.error} />}
            />
          ) : (
            <Button
              title={t.profile.signIn}
              onPress={() => router.push('/auth')}
              variant="primary"
              icon={<Ionicons name="log-in-outline" size={20} color={colors.white} />}
            />
          )}
        </View>
      )}

      <Text style={styles.sectionTitle}>Quick Access</Text>
      <Card style={styles.linksCard}>
        <MenuItem icon="settings-outline" label={t.profile.settings} color={colors.primary} onPress={() => router.push('/settings')} />
        <View style={styles.divider} />
        <MenuItem icon="time-outline" label={t.history.title} color={colors.secondaryDark} onPress={() => router.push('/history')} />
        <View style={styles.divider} />
        <MenuItem icon="book-outline" label={t.tabs.education} color={colors.accentDark} onPress={() => router.push('/education')} />
      </Card>

      <View style={styles.appInfo}>
        <Logo size="sm" horizontal />
        <Text style={styles.version}>{t.profile.appVersion}</Text>
      </View>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

function MenuItem({ icon, label, color, onPress }: { icon: keyof typeof Ionicons.glyphMap; label: string; color: string; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.menuItem} onPress={onPress} activeOpacity={0.7}>
      <View style={[styles.menuIconBox, { backgroundColor: color + '20' }]}>
        <Ionicons name={icon} size={20} color={color} />
      </View>
      <Text style={styles.menuLabel}>{label}</Text>
      <Ionicons name="chevron-forward" size={16} color={colors.textLight} />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  profileCard: { alignItems: 'center', paddingVertical: spacing.xl, marginBottom: spacing.lg, backgroundColor: colors.primary + '10' },
  avatarCircle: { width: 72, height: 72, borderRadius: 36, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md, ...shadows.md },
  avatarText: { fontSize: 24, fontWeight: '800', color: colors.white },
  roleLabel: { fontSize: 12, color: colors.primary, fontWeight: '600', marginBottom: 2 },
  signedInLabel: { fontSize: 12, color: colors.textSecondary, marginBottom: 2 },
  userId: { fontSize: 14, fontWeight: '600', color: colors.text, maxWidth: 220, textAlign: 'center' },
  guestLabel: { fontSize: 14, fontWeight: '700', color: colors.primaryDark, marginBottom: 2 },
  notSignedIn: { fontSize: 13, color: colors.textSecondary },
  statBadge: { marginTop: spacing.lg, alignItems: 'center', backgroundColor: colors.primary + '20', paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, borderRadius: borderRadius.pill },
  statNumber: { fontSize: 22, fontWeight: '800', color: colors.primaryDark },
  statLabel: { fontSize: 11, color: colors.textSecondary },
  section: { marginBottom: spacing.lg },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  linksCard: { paddingVertical: 0, overflow: 'hidden', marginBottom: spacing.lg },
  menuItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: spacing.md, paddingHorizontal: spacing.md },
  menuIconBox: { width: 36, height: 36, borderRadius: 10, alignItems: 'center', justifyContent: 'center', marginRight: spacing.md },
  menuLabel: { flex: 1, fontSize: 15, fontWeight: '500', color: colors.text },
  divider: { height: 1, backgroundColor: colors.border, marginLeft: 60 },
  appInfo: { alignItems: 'center', marginTop: spacing.lg, gap: 4 },
  version: { fontSize: 12, color: colors.textLight },
});
