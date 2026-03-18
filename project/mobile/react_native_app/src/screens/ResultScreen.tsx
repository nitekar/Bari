/**
 * ResultScreen
 * Displays the anemia screening prediction, confidence score,
 * dietary guidance, and medical referral recommendation.
 */

import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {RouteProp} from '@react-navigation/native';
import {RootStackParamList} from '../../App';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Result'>;
  route: RouteProp<RootStackParamList, 'Result'>;
};

/** Map severity to colour accent */
const SEVERITY_COLORS: Record<string, string> = {
  Normal: '#34a853',
  Mild: '#fbbc04',
  Moderate: '#fa7b17',
  Severe: '#ea4335',
};

/** Map urgency level to emoji badge */
const URGENCY_BADGE: Record<string, string> = {
  low: '🟢 Low',
  'low-medium': '🟡 Low–Medium',
  medium: '🟠 Medium',
  critical: '🔴 Critical',
};

export default function ResultScreen({navigation, route}: Props): React.JSX.Element {
  const {prediction} = route.params;
  const accentColor = SEVERITY_COLORS[prediction.diagnosis] ?? '#555';
  const confidencePct = Math.round(prediction.confidence * 100);
  const urgencyBadge = URGENCY_BADGE[prediction.urgency] ?? prediction.urgency;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* Diagnosis header */}
      <View style={[styles.diagnosisCard, {borderColor: accentColor}]}>
        <Text style={styles.diagnosisLabel}>Diagnosis</Text>
        <Text style={[styles.diagnosisValue, {color: accentColor}]}>
          {prediction.diagnosis}
        </Text>
        <Text style={styles.confidence}>
          Confidence: {confidencePct}%
        </Text>
        <Text style={styles.urgency}>Urgency: {urgencyBadge}</Text>
      </View>

      {/* Class probability bars */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Class Probabilities</Text>
        {Object.entries(prediction.class_probabilities).map(([cls, prob]) => {
          const pct = Math.round((prob as number) * 100);
          const barColor = SEVERITY_COLORS[cls] ?? '#ccc';
          return (
            <View key={cls} style={styles.probRow}>
              <Text style={styles.probLabel}>{cls}</Text>
              <View style={styles.probBarBg}>
                <View style={[styles.probBar, {width: `${pct}%`, backgroundColor: barColor}]} />
              </View>
              <Text style={styles.probPct}>{pct}%</Text>
            </View>
          );
        })}
      </View>

      {/* Nutritional guidance */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🥗 Nutritional Guidance</Text>
        <Text style={styles.bodyText}>{prediction.nutrition_advice}</Text>
      </View>

      {/* Recommended foods */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🍽 Recommended Foods</Text>
        {prediction.recommended_foods.map((food, idx) => (
          <Text key={idx} style={styles.bulletItem}>• {food}</Text>
        ))}
      </View>

      {/* Referral */}
      <View style={[styles.section, styles.referralCard]}>
        <Text style={styles.sectionTitle}>🏥 Medical Referral</Text>
        <Text style={styles.bodyText}>{prediction.referral_action}</Text>
      </View>

      {/* Restart */}
      <TouchableOpacity
        style={styles.buttonRestart}
        onPress={() => navigation.navigate('Capture')}>
        <Text style={styles.buttonText}>New Screening</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flexGrow: 1, padding: 20, backgroundColor: '#f5f7fa'},

  diagnosisCard: {
    borderWidth: 3, borderRadius: 12, padding: 20, marginBottom: 20,
    alignItems: 'center', backgroundColor: '#fff',
  },
  diagnosisLabel: {fontSize: 13, color: '#666', textTransform: 'uppercase', letterSpacing: 1},
  diagnosisValue: {fontSize: 32, fontWeight: 'bold', marginVertical: 4},
  confidence: {fontSize: 16, color: '#444'},
  urgency: {fontSize: 14, color: '#444', marginTop: 4},

  section: {
    backgroundColor: '#fff', borderRadius: 10, padding: 16,
    marginBottom: 16, shadowColor: '#000', shadowOpacity: 0.05,
    shadowRadius: 4, elevation: 2,
  },
  sectionTitle: {fontSize: 16, fontWeight: 'bold', color: '#1a1a2e', marginBottom: 10},
  bodyText: {fontSize: 14, color: '#444', lineHeight: 22},
  bulletItem: {fontSize: 14, color: '#444', lineHeight: 22, marginBottom: 4},

  probRow: {flexDirection: 'row', alignItems: 'center', marginBottom: 8},
  probLabel: {width: 80, fontSize: 13, color: '#333'},
  probBarBg: {flex: 1, height: 12, backgroundColor: '#e0e0e0', borderRadius: 6, overflow: 'hidden', marginHorizontal: 8},
  probBar: {height: '100%', borderRadius: 6},
  probPct: {width: 40, fontSize: 12, color: '#555', textAlign: 'right'},

  referralCard: {backgroundColor: '#fff3e0'},

  buttonRestart: {
    backgroundColor: '#1a73e8', paddingVertical: 14, paddingHorizontal: 40,
    borderRadius: 8, alignItems: 'center', marginTop: 8, marginBottom: 32,
  },
  buttonText: {color: '#fff', fontWeight: 'bold', fontSize: 16},
});
