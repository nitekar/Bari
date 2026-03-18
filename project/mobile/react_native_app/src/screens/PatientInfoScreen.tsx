/**
 * PatientInfoScreen
 * Collects patient age (months) and gender, then sends the request to the API.
 */

import React, {useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {RouteProp} from '@react-navigation/native';
import {RootStackParamList} from '../../App';
import {predictAnemia} from '../services/api';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'PatientInfo'>;
  route: RouteProp<RootStackParamList, 'PatientInfo'>;
};

export default function PatientInfoScreen({navigation, route}: Props): React.JSX.Element {
  const {imageUri} = route.params;

  const [age, setAge] = useState('');
  const [gender, setGender] = useState<'Male' | 'Female' | ''>('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    const ageNum = parseInt(age, 10);
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 1200) {
      Alert.alert('Invalid age', 'Please enter a valid age in months (0–1200).');
      return;
    }
    if (!gender) {
      Alert.alert('Missing gender', 'Please select the patient gender.');
      return;
    }

    setLoading(true);
    try {
      const result = await predictAnemia(imageUri, ageNum, gender);
      navigation.navigate('Result', {prediction: result});
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ||
        error?.message ||
        'Unable to reach the server. Please check your connection.';
      Alert.alert('Prediction Failed', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Step 2: Patient Information</Text>
      <Text style={styles.subtitle}>
        Enter the patient's basic information to improve prediction accuracy.
      </Text>

      {/* Age input */}
      <View style={styles.fieldWrapper}>
        <Text style={styles.label}>Age (months)</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          placeholder="e.g. 36"
          value={age}
          onChangeText={setAge}
          maxLength={4}
        />
      </View>

      {/* Gender selection */}
      <View style={styles.fieldWrapper}>
        <Text style={styles.label}>Gender</Text>
        <View style={styles.genderRow}>
          {(['Male', 'Female'] as const).map(g => (
            <TouchableOpacity
              key={g}
              style={[
                styles.genderButton,
                gender === g && styles.genderButtonSelected,
              ]}
              onPress={() => setGender(g)}>
              <Text
                style={[
                  styles.genderButtonText,
                  gender === g && styles.genderButtonTextSelected,
                ]}>
                {g === 'Male' ? '♂  Male' : '♀  Female'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Submit */}
      {loading ? (
        <ActivityIndicator size="large" color="#1a73e8" style={{marginTop: 24}} />
      ) : (
        <TouchableOpacity style={styles.buttonSubmit} onPress={handleSubmit}>
          <Text style={styles.buttonText}>Analyse →</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flexGrow: 1, alignItems: 'center', padding: 24, backgroundColor: '#f5f7fa'},
  title: {fontSize: 20, fontWeight: 'bold', color: '#1a1a2e', marginBottom: 8},
  subtitle: {fontSize: 14, color: '#555', textAlign: 'center', marginBottom: 24},
  fieldWrapper: {width: '90%', marginBottom: 20},
  label: {fontSize: 15, fontWeight: '600', color: '#333', marginBottom: 8},
  input: {
    borderWidth: 1, borderColor: '#c0c0c0', borderRadius: 8,
    padding: 12, fontSize: 16, backgroundColor: '#fff',
  },
  genderRow: {flexDirection: 'row', justifyContent: 'space-between'},
  genderButton: {
    flex: 1, marginHorizontal: 4, paddingVertical: 12, borderRadius: 8,
    borderWidth: 2, borderColor: '#c0c0c0', alignItems: 'center',
    backgroundColor: '#fff',
  },
  genderButtonSelected: {borderColor: '#1a73e8', backgroundColor: '#e8f0fe'},
  genderButtonText: {fontSize: 15, color: '#555', fontWeight: '600'},
  genderButtonTextSelected: {color: '#1a73e8'},
  buttonSubmit: {
    backgroundColor: '#1a73e8', paddingVertical: 14, paddingHorizontal: 40,
    borderRadius: 8, marginTop: 16, width: '80%', alignItems: 'center',
  },
  buttonText: {color: '#fff', fontWeight: 'bold', fontSize: 17},
});
