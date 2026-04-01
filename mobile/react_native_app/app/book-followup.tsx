import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity, Alert, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, typography, spacing, borderRadius } from '../src/shared/theme';
import { Card, Button } from '../src/shared/components';
import { useTranslation } from '../src/i18n';
import { useStore, Appointment } from '../src/store/useStore';

const CLINICS = [
  'King Faisal Hospital',
  'CHUK – University Teaching Hospital',
  'Rwanda Military Hospital',
  'Butaro District Hospital',
  'Kibagabaga Hospital',
];

export default function BookFollowupScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const addAppointment = useStore((s) => s.addAppointment);

  const [selectedClinic, setSelectedClinic] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedTime, setSelectedTime] = useState<string>('');
  const [notes, setNotes] = useState<string>('');

  const dates = [
    { label: 'Tomorrow', value: new Date(Date.now() + 86400000).toISOString().split('T')[0] },
    { label: 'In 3 Days', value: new Date(Date.now() + 3 * 86400000).toISOString().split('T')[0] },
    { label: 'Next Week', value: new Date(Date.now() + 7 * 86400000).toISOString().split('T')[0] },
  ];

  const times = ['09:00 AM', '11:00 AM', '02:00 PM', '04:00 PM'];

  const handleBooking = () => {
    if (!selectedClinic || !selectedDate || !selectedTime) {
      if (Platform.OS === 'web') {
        window.alert(t.bookFollowup.fillRequired);
      } else {
        Alert.alert('Error', t.bookFollowup.fillRequired);
      }
      return;
    }

    const newAppt: Appointment = {
      id: Math.random().toString(36).substr(2, 9),
      clinicName: selectedClinic,
      date: selectedDate,
      time: selectedTime,
      notes: notes,
      status: 'upcoming',
    };

    addAppointment(newAppt);

    if (Platform.OS === 'web') {
      window.alert(t.bookFollowup.bookingSuccessMsg);
      router.back();
    } else {
      Alert.alert(t.bookFollowup.bookingSuccess, t.bookFollowup.bookingSuccessMsg, [
        { text: 'OK', onPress: () => router.back() },
      ]);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Card style={styles.headerCard}>
        <Ionicons name="calendar" size={32} color={colors.primary} style={styles.headerIcon} />
        <Text style={styles.title}>{t.bookFollowup.title}</Text>
        <Text style={styles.description}>{t.bookFollowup.description}</Text>
      </Card>

      <Card>
        <Text style={styles.label}>{t.bookFollowup.selectClinic}</Text>
        {CLINICS.map((clinic, index) => (
          <TouchableOpacity
            key={index}
            style={[styles.optionItem, selectedClinic === clinic && styles.optionItemSelected]}
            onPress={() => setSelectedClinic(clinic)}
          >
            <Ionicons name="medical" size={20} color={selectedClinic === clinic ? colors.primary : colors.textSecondary} />
            <Text style={[styles.optionText, selectedClinic === clinic && styles.optionTextSelected]}>{clinic}</Text>
          </TouchableOpacity>
        ))}
      </Card>

      <Card>
        <Text style={styles.label}>{t.bookFollowup.selectDate}</Text>
        <View style={styles.pillContainer}>
          {dates.map((dateObj, index) => (
            <TouchableOpacity
              key={index}
              style={[styles.pill, selectedDate === dateObj.value && styles.pillSelected]}
              onPress={() => setSelectedDate(dateObj.value)}
            >
              <Text style={[styles.pillText, selectedDate === dateObj.value && styles.pillTextSelected]}>
                {dateObj.label}
              </Text>
              <Text style={[styles.pillSubtext, selectedDate === dateObj.value && styles.pillTextSelected]}>
                {dateObj.value}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      <Card>
        <Text style={styles.label}>{t.bookFollowup.selectTime}</Text>
        <View style={styles.pillContainer}>
          {times.map((time, index) => (
            <TouchableOpacity
              key={index}
              style={[styles.pill, selectedTime === time && styles.pillSelected]}
              onPress={() => setSelectedTime(time)}
            >
              <Text style={[styles.pillText, selectedTime === time && styles.pillTextSelected]}>{time}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      <Card>
        <Text style={styles.label}>{t.bookFollowup.notesOptions}</Text>
        <TextInput
          style={styles.input}
          placeholder={t.bookFollowup.notesPlaceholder}
          placeholderTextColor={colors.textLight}
          value={notes}
          onChangeText={setNotes}
          multiline
          numberOfLines={3}
        />
      </Card>

      <Button
        title={t.bookFollowup.confirmBooking}
        onPress={handleBooking}
        variant="primary"
        style={styles.bookButton}
        icon={<Ionicons name="checkmark-circle-outline" size={20} color={colors.white} />}
      />
      <View style={{ height: spacing.xxl }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.md,
  },
  headerCard: {
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  headerIcon: {
    marginBottom: spacing.sm,
  },
  title: {
    ...typography.title,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  description: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
  },
  label: {
    ...typography.subtitle,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  optionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.sm,
    backgroundColor: colors.surface,
  },
  optionItemSelected: {
    borderColor: colors.primary,
    backgroundColor: colors.primaryLight,
  },
  optionText: {
    ...typography.body,
    color: colors.text,
    marginLeft: spacing.sm,
  },
  optionTextSelected: {
    fontWeight: 'bold',
    color: colors.primaryDark,
  },
  pillContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  pill: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    alignItems: 'center',
  },
  pillSelected: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  pillText: {
    ...typography.bodyBold,
    color: colors.text,
  },
  pillSubtext: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  pillTextSelected: {
    color: colors.white,
  },
  input: {
    ...typography.body,
    color: colors.text,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    backgroundColor: colors.surface,
    textAlignVertical: 'top',
    minHeight: 100,
  },
  bookButton: {
    marginTop: spacing.md,
    marginBottom: spacing.xl,
  },
});
