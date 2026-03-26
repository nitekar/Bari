/**
 * app/parent-sleep.tsx — Sleep entry form + log list
 */
import React, { useState, useMemo } from 'react';
import {
  ScrollView, View, Text, StyleSheet, TextInput,
  TouchableOpacity, Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../src/shared/theme';
import { Card, Button } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import type { SleepLog } from '../src/store/useStore';

function parseHHMM(str: string): number | null {
  const [hStr, mStr] = str.split(':');
  const h = parseInt(hStr, 10);
  const m = parseInt(mStr ?? '0', 10);
  if (isNaN(h) || isNaN(m) || h < 0 || h > 23 || m < 0 || m > 59) return null;
  return h * 60 + m;
}

function calcDuration(start: string, end: string): number | null {
  const s = parseHHMM(start);
  const e = parseHHMM(end);
  if (s === null || e === null) return null;
  let diff = e - s;
  if (diff < 0) diff += 24 * 60; // crosses midnight
  return diff / 60;
}

// Age-appropriate sleep recommendations
function sleepRec(ageMonths: number | null): string {
  if (ageMonths === null) return '';
  if (ageMonths < 4) return 'Newborns need 14–17 hours of sleep per day.';
  if (ageMonths < 12) return 'Infants (4–11 months) need 12–15 hours per day.';
  if (ageMonths < 24) return 'Toddlers (1–2 years) need 11–14 hours per day.';
  if (ageMonths < 60) return 'Preschoolers (3–5 years) need 10–13 hours per day.';
  return 'School-age children need 9–11 hours per day.';
}

export default function ParentSleepScreen() {
  const addSleepLog = useStore((s) => s.addSleepLog);
  const sleepLogs = useStore((s) => s.sleepLogs);
  const childAgeMonths = useStore((s) => s.childAgeMonths);

  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [notes, setNotes] = useState('');

  const duration = useMemo(() => calcDuration(startTime, endTime), [startTime, endTime]);

  const handleAdd = () => {
    if (!startTime.trim() || !endTime.trim()) {
      Alert.alert('Missing fields', 'Please enter both start and end time.');
      return;
    }
    if (duration === null) {
      Alert.alert('Invalid time', 'Please use HH:MM format (e.g. 08:30).');
      return;
    }
    const today = new Date().toISOString().split('T')[0];
    const log: SleepLog = {
      id: Date.now().toString(),
      startTime: startTime.trim(),
      endTime: endTime.trim(),
      durationHours: parseFloat(duration.toFixed(2)),
      notes: notes.trim(),
      date: today,
    };
    addSleepLog(log);
    setStartTime('');
    setEndTime('');
    setNotes('');
  };

  const rec = sleepRec(childAgeMonths);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
      {/* Recommendation */}
      {rec && (
        <Card style={styles.recCard}>
          <View style={styles.recRow}>
            <Ionicons name="moon" size={18} color={colors.primaryDark} />
            <Text style={styles.recText}>{rec}</Text>
          </View>
        </Card>
      )}

      {/* Add entry form */}
      <Card style={styles.formCard}>
        <Text style={styles.formTitle}>Log Sleep Session</Text>

        <Text style={styles.fieldLabel}>Start Time (HH:MM)</Text>
        <TextInput
          style={styles.input}
          value={startTime}
          onChangeText={setStartTime}
          placeholder="e.g. 20:30"
          placeholderTextColor={colors.textLight}
          keyboardType="numbers-and-punctuation"
        />

        <Text style={styles.fieldLabel}>End Time (HH:MM)</Text>
        <TextInput
          style={styles.input}
          value={endTime}
          onChangeText={setEndTime}
          placeholder="e.g. 06:30"
          placeholderTextColor={colors.textLight}
          keyboardType="numbers-and-punctuation"
        />

        {duration !== null && (
          <View style={styles.durationRow}>
            <Ionicons name="time-outline" size={16} color={colors.primary} />
            <Text style={styles.durationText}>Duration: {duration.toFixed(1)} hours</Text>
          </View>
        )}

        <Text style={styles.fieldLabel}>Notes (optional)</Text>
        <TextInput
          style={[styles.input, styles.notesInput]}
          value={notes}
          onChangeText={setNotes}
          placeholder="e.g. woke up twice"
          placeholderTextColor={colors.textLight}
          multiline
        />

        <View style={{ marginTop: spacing.md }}>
          <Button
            title="Add Sleep Entry"
            onPress={handleAdd}
            variant="primary"
            icon={<Ionicons name="add-circle-outline" size={18} color={colors.white} />}
          />
        </View>
      </Card>

      {/* Log list */}
      {sleepLogs.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recent Entries</Text>
          {sleepLogs.slice(0, 20).map(log => (
            <Card key={log.id} style={styles.logCard}>
              <View style={styles.logRow}>
                <View style={styles.logLeft}>
                  <Text style={styles.logDuration}>{log.durationHours.toFixed(1)}h</Text>
                  <Text style={styles.logTime}>{log.startTime} – {log.endTime}</Text>
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.logDate}>{log.date}</Text>
                  {log.notes ? <Text style={styles.logNotes}>{log.notes}</Text> : null}
                </View>
              </View>
            </Card>
          ))}
        </>
      )}

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg },
  recCard: { backgroundColor: colors.primary + '12', marginBottom: spacing.md },
  recRow: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm },
  recText: { flex: 1, fontSize: 13, color: colors.primaryDark, lineHeight: 20, fontWeight: '500' },
  formCard: { marginBottom: spacing.lg },
  formTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: spacing.md },
  fieldLabel: { fontSize: 12, fontWeight: '600', color: colors.textSecondary, marginBottom: 6, marginTop: spacing.sm },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    fontSize: 15, color: colors.text,
  },
  notesInput: { minHeight: 64, textAlignVertical: 'top' },
  durationRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginTop: spacing.sm, paddingVertical: spacing.xs, paddingHorizontal: spacing.sm, backgroundColor: colors.primary + '12', borderRadius: borderRadius.sm },
  durationText: { fontSize: 13, fontWeight: '600', color: colors.primary },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  logCard: { marginBottom: spacing.sm, paddingVertical: spacing.sm },
  logRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  logLeft: { alignItems: 'center', width: 56 },
  logDuration: { fontSize: 18, fontWeight: '800', color: colors.primaryDark },
  logTime: { fontSize: 10, color: colors.textLight },
  logDate: { fontSize: 13, fontWeight: '600', color: colors.text },
  logNotes: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
});
