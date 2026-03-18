/**
 * PatientFormScreen.js
 * --------------------
 * Patient demographic input form.
 * Collects age and gender, then triggers the API prediction.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { predictAnemia } from './api';

export default function PatientFormScreen({ route, navigation }) {
  const { imageUri } = route.params;

  const [age, setAge] = useState('');
  const [gender, setGender] = useState(null); // 'Male' | 'Female'
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    // Validate
    const ageNum = parseInt(age, 10);
    if (!age || isNaN(ageNum) || ageNum < 0 || ageNum > 120) {
      Alert.alert('Invalid Age', 'Please enter a valid age between 0 and 120.');
      return;
    }
    if (!gender) {
      Alert.alert('Gender Required', 'Please select a gender.');
      return;
    }

    setLoading(true);
    try {
      const result = await predictAnemia(imageUri, ageNum, gender);
      navigation.navigate('Result', { result });
    } catch (err) {
      Alert.alert('Prediction Failed', err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Patient Information</Text>
        <Text style={styles.subtitle}>
          Enter the patient's demographic details to complete the anemia screening.
        </Text>

        {/* Age Field */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Age (years)</Text>
          <TextInput
            style={styles.input}
            value={age}
            onChangeText={setAge}
            keyboardType="numeric"
            placeholder="e.g. 28"
            placeholderTextColor="#888"
            maxLength={3}
          />
        </View>

        {/* Gender Field */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Gender</Text>
          <View style={styles.genderRow}>
            {['Male', 'Female'].map((g) => (
              <TouchableOpacity
                key={g}
                style={[styles.genderOption, gender === g && styles.genderSelected]}
                onPress={() => setGender(g)}
              >
                <Text style={[styles.genderText, gender === g && styles.genderTextSelected]}>
                  {g}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Submit */}
        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleSubmit}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Run Screening</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.backLink}
          onPress={() => navigation.goBack()}
          disabled={loading}
        >
          <Text style={styles.backLinkText}>← Retake image</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#1a1a2e',
    padding: 24,
    justifyContent: 'center',
  },
  title: { color: '#fff', fontSize: 22, fontWeight: '700', marginBottom: 6, textAlign: 'center' },
  subtitle: { color: '#aaa', fontSize: 13, marginBottom: 28, textAlign: 'center' },
  fieldGroup: { marginBottom: 20 },
  label: { color: '#ccc', fontSize: 14, fontWeight: '600', marginBottom: 8 },
  input: {
    backgroundColor: '#2d2d44',
    color: '#fff',
    borderRadius: 10,
    padding: 14,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#444',
  },
  genderRow: { flexDirection: 'row', gap: 12 },
  genderOption: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#444',
    alignItems: 'center',
    backgroundColor: '#2d2d44',
  },
  genderSelected: { borderColor: '#e74c3c', backgroundColor: '#3d1a1a' },
  genderText: { color: '#aaa', fontSize: 15, fontWeight: '600' },
  genderTextSelected: { color: '#e74c3c' },
  button: {
    backgroundColor: '#e74c3c',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 12,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
  backLink: { alignItems: 'center', marginTop: 18 },
  backLinkText: { color: '#888', fontSize: 14 },
});
