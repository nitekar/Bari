/**
 * app/parent-feeding.tsx — Feeding entry form + log list
 */
import React, { useState } from 'react';
import {
  ScrollView, View, Text, StyleSheet, TextInput,
  TouchableOpacity, Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../src/shared/theme';
import { Card, Button } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import type { FeedingLog } from '../src/store/useStore';

type FeedingType = 'breastfeeding' | 'formula' | 'solid';

const FEEDING_TYPES: { type: FeedingType; label: string; icon: any; color: string }[] = [
  { type: 'breastfeeding', label: 'Breastfeeding', icon: 'heart-outline', color: colors.secondary },
  { type: 'formula',       label: 'Formula',       icon: 'flask-outline', color: colors.primary },
  { type: 'solid',         label: 'Solid Food',    icon: 'restaurant-outline', color: colors.accentDark },
];

// Iron-rich food suggestions for parents who rarely log solids
const IRON_SUGGESTIONS = [
  'Pureed liver (chicken or beef) — excellent heme iron source',
  'Mashed beans or lentils with tomato for vitamin C',
  'Moringa leaf powder mixed into porridge',
  'Mashed egg yolk — iron + protein combo',
  'Iron-fortified infant cereal',
];

export default function ParentFeedingScreen() {
  const addFeedingLog = useStore((s) => s.addFeedingLog);
  const feedingLogs = useStore((s) => s.feedingLogs);

  const [feedingType, setFeedingType] = useState<FeedingType>('breastfeeding');
  const [time, setTime] = useState('');
  const [quantityMl, setQuantityMl] = useState('');
  const [notes, setNotes] = useState('');

  const handleAdd = () => {
    if (!time.trim()) {
      Alert.alert('Missing time', 'Please enter the feeding time (HH:MM).');
      return;
    }
    const log: FeedingLog = {
      id: Date.now().toString(),
      type: feedingType,
      time: time.trim(),
      quantityMl: quantityMl ? parseFloat(quantityMl) : undefined,
      notes: notes.trim(),
      date: new Date().toISOString().split('T')[0],
    };
    addFeedingLog(log);
    setTime('');
    setQuantityMl('');
    setNotes('');
  };

  // Check if solid frequency is low (fewer than 3 solid logs in the last 7 days)
  const last7days = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  const recentSolids = feedingLogs.filter(l => l.type === 'solid' && l.date >= last7days).length;
  const showIronSuggestion = recentSolids < 3 && feedingLogs.length >= 3;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
      {/* Iron suggestion banner */}
      {showIronSuggestion && (
        <Card style={styles.tipCard}>
          <View style={styles.tipHeader}>
            <Ionicons name="leaf-outline" size={18} color={colors.accentDark} />
            <Text style={styles.tipTitle}>Iron-Rich Food Ideas</Text>
          </View>
          {IRON_SUGGESTIONS.slice(0, 3).map((s, i) => (
            <View key={i} style={styles.tipRow}>
              <Text style={styles.tipBullet}>·</Text>
              <Text style={styles.tipText}>{s}</Text>
            </View>
          ))}
        </Card>
      )}

      {/* Form */}
      <Card style={styles.formCard}>
        <Text style={styles.formTitle}>Log Feeding</Text>

        {/* Type selector pills */}
        <Text style={styles.fieldLabel}>Feeding Type</Text>
        <View style={styles.typeRow}>
          {FEEDING_TYPES.map(opt => (
            <TouchableOpacity
              key={opt.type}
              style={[styles.typePill, feedingType === opt.type && { backgroundColor: opt.color, borderColor: opt.color }]}
              onPress={() => setFeedingType(opt.type)}
              activeOpacity={0.8}
            >
              <Ionicons name={opt.icon} size={14} color={feedingType === opt.type ? colors.white : colors.text} />
              <Text style={[styles.typePillText, feedingType === opt.type && { color: colors.white }]}>
                {opt.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.fieldLabel}>Time (HH:MM)</Text>
        <TextInput
          style={styles.input}
          value={time}
          onChangeText={setTime}
          placeholder="e.g. 14:30"
          placeholderTextColor={colors.textLight}
          keyboardType="numbers-and-punctuation"
        />

        {feedingType === 'formula' && (
          <>
            <Text style={styles.fieldLabel}>Quantity (ml, optional)</Text>
            <TextInput
              style={styles.input}
              value={quantityMl}
              onChangeText={setQuantityMl}
              placeholder="e.g. 120"
              placeholderTextColor={colors.textLight}
              keyboardType="decimal-pad"
            />
          </>
        )}

        <Text style={styles.fieldLabel}>Notes (optional)</Text>
        <TextInput
          style={[styles.input, styles.notesInput]}
          value={notes}
          onChangeText={setNotes}
          placeholder="e.g. ate well, tried mashed banana"
          placeholderTextColor={colors.textLight}
          multiline
        />

        <View style={{ marginTop: spacing.md }}>
          <Button
            title="Add Feeding Entry"
            onPress={handleAdd}
            variant="primary"
            icon={<Ionicons name="add-circle-outline" size={18} color={colors.white} />}
          />
        </View>
      </Card>

      {/* Recent entries */}
      {feedingLogs.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recent Entries</Text>
          {feedingLogs.slice(0, 20).map(log => {
            const typeInfo = FEEDING_TYPES.find(t => t.type === log.type)!;
            return (
              <Card key={log.id} style={styles.logCard}>
                <View style={styles.logRow}>
                  <View style={[styles.logIcon, { backgroundColor: typeInfo.color + '20' }]}>
                    <Ionicons name={typeInfo.icon} size={18} color={typeInfo.color} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.logType}>{typeInfo.label}</Text>
                    <Text style={styles.logTime}>{log.time} · {log.date}</Text>
                    {log.quantityMl && <Text style={styles.logQty}>{log.quantityMl} ml</Text>}
                    {log.notes ? <Text style={styles.logNotes}>{log.notes}</Text> : null}
                  </View>
                </View>
              </Card>
            );
          })}
        </>
      )}

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  tipCard: { backgroundColor: colors.accentDark + '10', marginBottom: spacing.md },
  tipHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.sm },
  tipTitle: { fontSize: 14, fontWeight: '700', color: colors.accentDark },
  tipRow: { flexDirection: 'row', gap: spacing.xs, marginBottom: 3 },
  tipBullet: { fontSize: 14, color: colors.accentDark, lineHeight: 20 },
  tipText: { flex: 1, fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
  formCard: { marginBottom: spacing.lg },
  formTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: spacing.md },
  fieldLabel: { fontSize: 12, fontWeight: '600', color: colors.textSecondary, marginBottom: 6, marginTop: spacing.sm },
  typeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  typePill: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: spacing.sm, paddingVertical: 8,
    borderRadius: borderRadius.pill, borderWidth: 1,
    borderColor: colors.border, backgroundColor: colors.surface,
  },
  typePillText: { fontSize: 12, fontWeight: '600', color: colors.text },
  input: {
    backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.md, paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm, fontSize: 15, color: colors.text,
  },
  notesInput: { minHeight: 64, textAlignVertical: 'top' },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  logCard: { marginBottom: spacing.sm, paddingVertical: spacing.sm },
  logRow: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.md },
  logIcon: { width: 40, height: 40, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  logType: { fontSize: 14, fontWeight: '600', color: colors.text },
  logTime: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  logQty: { fontSize: 12, color: colors.primary, marginTop: 2 },
  logNotes: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
});
