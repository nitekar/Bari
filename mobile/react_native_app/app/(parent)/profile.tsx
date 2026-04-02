/**
 * app/(parent)/profile.tsx — Child profile card + sign-out
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  Platform,
  TextInput,
  TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../src/shared/theme';
import { Card, Button } from '../../src/shared/components';
import { useStore } from '../../src/store/useStore';
import { signOut } from '../../src/services/supabaseAuth';
import { isSupabaseConfigured } from '../../src/services/supabase';
import { useTranslation } from '../../src/i18n';

type Gender = 'male' | 'female';

export default function ParentProfileScreen() {
  const router = useRouter();
  const { t } = useTranslation();

  const userId = useStore((s) => s.userId);
  const childName = useStore((s) => s.childName);
  const childAgeMonths = useStore((s) => s.childAgeMonths);
  const childGender = useStore((s) => s.childGender);
  const setUserId = useStore((s) => s.setUserId);
  const setChildProfile = useStore((s) => s.setChildProfile);

  const [editName, setEditName] = useState(childName ?? '');
  const [editAge, setEditAge] = useState(childAgeMonths != null ? String(childAgeMonths) : '');
  const [editGender, setEditGender] = useState<Gender>(childGender ?? 'female');
  const [signingOut, setSigningOut] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    const ageNum = parseInt(editAge, 10);
    setChildProfile(
      editName.trim() || null,
      isNaN(ageNum) ? null : ageNum,
      editGender,
    );
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

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

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {/* Child profile card */}
      <Card style={styles.childCard}>
        <View style={styles.childAvatar}>
          <Ionicons name={editGender === 'male' ? 'happy' : 'happy-outline'} size={32} color={colors.secondary} />
        </View>
        <Text style={styles.cardTitle}>Child Profile</Text>

        <Text style={styles.fieldLabel}>Child's Name</Text>
        <TextInput
          style={styles.input}
          value={editName}
          onChangeText={setEditName}
          placeholder="e.g. Amara"
          placeholderTextColor={colors.textLight}
        />

        <Text style={styles.fieldLabel}>Age (months)</Text>
        <TextInput
          style={styles.input}
          value={editAge}
          onChangeText={setEditAge}
          placeholder="e.g. 18"
          keyboardType="number-pad"
          placeholderTextColor={colors.textLight}
        />

        <Text style={styles.fieldLabel}>Gender</Text>
        <View style={styles.genderRow}>
          {(['female', 'male'] as Gender[]).map((g) => (
            <TouchableOpacity
              key={g}
              style={[styles.genderBtn, editGender === g && styles.genderBtnActive]}
              onPress={() => setEditGender(g)}
              activeOpacity={0.8}
            >
              <Ionicons
                name={g === 'female' ? 'female-outline' : 'male-outline'}
                size={18}
                color={editGender === g ? colors.white : colors.text}
              />
              <Text style={[styles.genderBtnText, editGender === g && styles.genderBtnTextActive]}>
                {g.charAt(0).toUpperCase() + g.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={{ marginTop: spacing.md }}>
          <Button
            title={saved ? 'Saved!' : 'Save Profile'}
            onPress={handleSave}
            variant={saved ? 'secondary' : 'primary'}
            icon={<Ionicons name={saved ? 'checkmark-circle' : 'save-outline'} size={18} color={saved ? colors.primary : colors.white} />}
          />
        </View>
      </Card>

      {/* Sign out */}
      {isSupabaseConfigured && userId && (
        <Button
          title={signingOut ? t.profile.signingOut : t.profile.signOut}
          onPress={handleSignOut}
          variant="outline"
          loading={signingOut}
          icon={<Ionicons name="log-out-outline" size={20} color={colors.error} />}
        />
      )}

      {!userId && isSupabaseConfigured && (
        <View style={styles.guestActionContainer}>
          <Text style={styles.guestActionText}>Register to save records, track development, and get personalized guidance.</Text>
          <Button
            title="Get Started"
            onPress={() => router.push('/auth')}
            variant="primary"
            icon={<Ionicons name="sparkles-outline" size={20} color={colors.white} />}
          />
        </View>
      )}

      {/* ── Legal ── */}
      <TouchableOpacity
        style={styles.legalBtn}
        onPress={() => router.push('/legal')}
        activeOpacity={0.7}
      >
        <Ionicons name="shield-checkmark-outline" size={18} color="#7E57C2" />
        <Text style={styles.legalText}>EULA & Privacy Policy</Text>
        <Ionicons name="chevron-forward" size={16} color={colors.textLight} />
      </TouchableOpacity>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  childCard: { marginBottom: spacing.lg, backgroundColor: colors.secondary + '08' },
  childAvatar: { alignSelf: 'center', width: 72, height: 72, borderRadius: 36, backgroundColor: colors.secondary + '20', alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md },
  cardTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: spacing.md, textAlign: 'center' },
  fieldLabel: { fontSize: 12, fontWeight: '600', color: colors.textSecondary, marginBottom: 6, marginTop: spacing.sm },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 15,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  genderRow: { flexDirection: 'row', gap: spacing.sm },
  genderBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 6, paddingVertical: spacing.sm, borderRadius: borderRadius.md,
    borderWidth: 1, borderColor: colors.border, backgroundColor: colors.surface,
  },
  genderBtnActive: { backgroundColor: colors.secondary, borderColor: colors.secondary },
  genderBtnText: { fontSize: 14, fontWeight: '600', color: colors.text },
  genderBtnTextActive: { color: colors.white },
  legalBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.lg,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  legalText: { flex: 1, fontSize: 14, fontWeight: '500', color: colors.text },
  guestActionContainer: { marginTop: spacing.md, paddingVertical: spacing.md },
  guestActionText: { fontSize: 13, color: colors.textSecondary, textAlign: 'center', marginBottom: spacing.md, lineHeight: 20 },
});
