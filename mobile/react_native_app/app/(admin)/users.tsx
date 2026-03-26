/**
 * app/(admin)/users.tsx — User management list
 * Fetches from profiles table; handles RLS restriction gracefully.
 */
import React, { useEffect, useState } from 'react';
import { ScrollView, View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../src/shared/theme';
import { Card } from '../../src/shared/components';
import { supabase } from '../../src/services/supabase';
import type { UserRole } from '../../src/services/supabaseAuth';

interface UserRow {
  id: string;
  role: UserRole;
  full_name: string | null;
}

const ROLE_COLORS: Record<UserRole, string> = {
  admin: '#7E57C2',
  chw: colors.primary,
  parent: colors.secondary,
};

function initials(name: string | null, id: string) {
  if (name && name.trim()) return name.trim().slice(0, 2).toUpperCase();
  return id.slice(0, 2).toUpperCase();
}

export default function UsersScreen() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    supabase
      .from('profiles')
      .select('id, role, full_name')
      .order('role')
      .then(({ data, error: err }) => {
        if (err) {
          setError(
            err.code === '42501'
              ? 'Access denied. Only admins can view all users.'
              : err.message,
          );
        } else {
          setUsers((data as UserRow[]) ?? []);
        }
        setLoading(false);
      });
  }, []);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Users</Text>
        {!loading && !error && <Text style={styles.count}>{users.length} registered</Text>}
      </View>

      {loading && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      )}

      {error && (
        <Card style={styles.errorCard}>
          <Ionicons name="lock-closed-outline" size={24} color={colors.error} />
          <Text style={styles.errorText}>{error}</Text>
        </Card>
      )}

      {!loading && !error && users.length === 0 && (
        <View style={styles.center}>
          <Ionicons name="people-outline" size={48} color={colors.textLight} />
          <Text style={styles.emptyText}>No users found</Text>
        </View>
      )}

      {users.map((u) => (
        <Card key={u.id} style={styles.card}>
          <View style={styles.row}>
            <View style={[styles.avatar, { backgroundColor: (ROLE_COLORS[u.role] ?? colors.primary) + '20' }]}>
              <Text style={[styles.avatarText, { color: ROLE_COLORS[u.role] ?? colors.primary }]}>
                {initials(u.full_name, u.id)}
              </Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.name}>{u.full_name || 'Unnamed User'}</Text>
              <Text style={styles.uid} numberOfLines={1}>{u.id}</Text>
            </View>
            <View style={[styles.badge, { backgroundColor: (ROLE_COLORS[u.role] ?? colors.primary) + '20' }]}>
              <Text style={[styles.badgeText, { color: ROLE_COLORS[u.role] ?? colors.primary }]}>
                {u.role}
              </Text>
            </View>
          </View>
        </Card>
      ))}

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.lg },
  title: { fontSize: 22, fontWeight: '800', color: colors.text },
  count: { fontSize: 13, color: colors.textSecondary },
  center: { alignItems: 'center', paddingVertical: spacing.xxl },
  emptyText: { fontSize: 16, color: colors.textSecondary, marginTop: spacing.md },
  errorCard: { alignItems: 'center', paddingVertical: spacing.xl, backgroundColor: colors.error + '10' },
  errorText: { fontSize: 14, color: colors.error, marginTop: spacing.sm, textAlign: 'center' },
  card: { marginBottom: spacing.sm, paddingVertical: spacing.sm },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  avatar: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  avatarText: { fontSize: 15, fontWeight: '800' },
  name: { fontSize: 15, fontWeight: '600', color: colors.text },
  uid: { fontSize: 11, color: colors.textLight, marginTop: 2 },
  badge: { paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: borderRadius.pill },
  badgeText: { fontSize: 11, fontWeight: '700' },
});
