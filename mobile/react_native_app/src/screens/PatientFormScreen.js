/**
 * PatientFormScreen
 * ─────────────────
 * Step 2 of the screening workflow.
 * Collects:
 *   • Age in months (numeric input)
 *   • Gender (Male / Female toggle)
 *
 * On submit, sends a multipart/form-data POST to the FastAPI /predict endpoint
 * and navigates to ResultScreen with the response.
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  ActivityIndicator,
  Image,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import axios from 'axios';

/** Change to your server address when testing on a physical device. */
const API_BASE_URL = 'http://10.0.2.2:8000'; // Android emulator → host machine

export default function PatientFormScreen({route, navigation}) {
  const {imageUri} = route.params;

  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [loading, setLoading] = useState(false);

  /* ── Helpers ───────────────────────────────────────────────────────────── */
  const validate = () => {
    const ageNum = parseInt(age, 10);
    if (!age || isNaN(ageNum) || ageNum < 0 || ageNum > 1200) {
      Alert.alert('Invalid Age', 'Please enter age in months (0 – 1200).');
      return false;
    }
    if (!gender) {
      Alert.alert('Gender Required', 'Please select Male or Female.');
      return false;
    }
    return true;
  };

  /* ── API call ──────────────────────────────────────────────────────────── */
  const submitPrediction = async () => {
    if (!validate()) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'conjunctiva.jpg',
      });
      formData.append('age', parseInt(age, 10));
      formData.append('gender', gender);

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {'Content-Type': 'multipart/form-data'},
        timeout: 30000,
      });

      navigation.navigate('Result', {
        prediction: response.data,
        imageUri,
        patientAge: age,
        patientGender: gender,
      });
    } catch (error) {
      const msg =
        error.response?.data?.detail ||
        error.message ||
        'Server unreachable. Check the API URL and your network.';
      Alert.alert('Prediction Error', String(msg));
    } finally {
      setLoading(false);
    }
  };

  /* ── Render ────────────────────────────────────────────────────────────── */
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        {/* Image thumbnail */}
        <Image source={{uri: imageUri}} style={styles.thumbnail} resizeMode="cover" />

        <Text style={styles.title}>Patient Information</Text>
        <Text style={styles.subtitle}>
          Enter the patient's basic details to complete the screening assessment.
        </Text>

        {/* Age input */}
        <View style={styles.field}>
          <Text style={styles.label}>Age (in months) *</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 24"
            keyboardType="numeric"
            value={age}
            onChangeText={setAge}
            maxLength={4}
            returnKeyType="done"
          />
          {age ? (
            <Text style={styles.hint}>
              ≈ {(parseInt(age, 10) / 12).toFixed(1)} years
            </Text>
          ) : null}
        </View>

        {/* Gender toggle */}
        <View style={styles.field}>
          <Text style={styles.label}>Gender *</Text>
          <View style={styles.genderRow}>
            {['Male', 'Female'].map(g => (
              <TouchableOpacity
                key={g}
                style={[styles.genderBtn, gender === g && styles.genderBtnActive]}
                onPress={() => setGender(g)}>
                <Text style={[styles.genderText, gender === g && styles.genderTextActive]}>
                  {g === 'Male' ? '♂ Male' : '♀ Female'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            ⚕️ This tool is a screening aid only. Always consult a qualified
            healthcare professional for diagnosis and treatment decisions.
          </Text>
        </View>

        {/* Submit */}
        {loading ? (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#1a237e" />
            <Text style={styles.loadingText}>Analysing image…</Text>
          </View>
        ) : (
          <TouchableOpacity
            style={[styles.submitBtn, (!age || !gender) && styles.submitBtnDisabled]}
            onPress={submitPrediction}
            disabled={!age || !gender}>
            <Text style={styles.submitText}>🔍  Analyse & Get Diagnosis</Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

/* ── Styles ──────────────────────────────────────────────────────────────── */
const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f5f5f5'},
  scroll: {padding: 20, alignItems: 'center'},

  thumbnail: {
    width: 100,
    height: 100,
    borderRadius: 10,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: '#1a237e',
  },

  title: {fontSize: 22, fontWeight: 'bold', color: '#1a237e', marginBottom: 4},
  subtitle: {fontSize: 14, color: '#555', textAlign: 'center', marginBottom: 24, lineHeight: 20},

  field: {width: '100%', marginBottom: 20},
  label: {fontSize: 15, fontWeight: '600', color: '#333', marginBottom: 8},
  input: {
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: '#c5cae9',
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    color: '#222',
  },
  hint: {fontSize: 12, color: '#888', marginTop: 4, marginLeft: 4},

  genderRow: {flexDirection: 'row', gap: 12},
  genderBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#c5cae9',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  genderBtnActive: {borderColor: '#1a237e', backgroundColor: '#e8eaf6'},
  genderText: {fontSize: 15, color: '#888', fontWeight: '600'},
  genderTextActive: {color: '#1a237e'},

  disclaimer: {
    backgroundColor: '#fff3e0',
    borderRadius: 10,
    padding: 14,
    width: '100%',
    marginBottom: 24,
  },
  disclaimerText: {fontSize: 12, color: '#e65100', lineHeight: 18},

  submitBtn: {
    backgroundColor: '#1a237e',
    paddingVertical: 16,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
  },
  submitBtnDisabled: {backgroundColor: '#9fa8da'},
  submitText: {color: '#fff', fontWeight: 'bold', fontSize: 16},

  loadingBox: {alignItems: 'center', marginVertical: 20},
  loadingText: {marginTop: 10, fontSize: 14, color: '#555'},
});
