/**
 * ResultScreen.js
 * ---------------
 * Displays the anemia screening result including:
 *   - Diagnosis label with colour-coded severity badge
 *   - Confidence bar
 *   - Recommended foods list
 *   - Referral action / alert
 */

import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';

const SEVERITY_COLORS = {
  Normal: '#27ae60',
  Mild: '#f39c12',
  Moderate: '#e67e22',
  Severe: '#e74c3c',
};

export default function ResultScreen({ route, navigation }) {
  const { result } = route.params;
  const {
    diagnosis,
    confidence,
    nutrition_advice,
    recommended_foods,
    referral_action,
  } = result;

  const severityColor = SEVERITY_COLORS[diagnosis] || '#95a5a6';
  const confidencePct = Math.round(confidence * 100);

  const handleNewScreening = () => {
    navigation.navigate('Capture');
  };

  // Alert the user if referral is urgent
  React.useEffect(() => {
    if (diagnosis === 'Severe') {
      Alert.alert(
        '⚠️ Urgent Referral Required',
        referral_action,
        [{ text: 'Understood', style: 'destructive' }]
      );
    }
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* Header */}
      <Text style={styles.title}>Screening Result</Text>

      {/* Diagnosis Badge */}
      <View style={[styles.badge, { backgroundColor: severityColor }]}>
        <Text style={styles.badgeText}>{diagnosis}</Text>
      </View>

      {/* Confidence Bar */}
      <View style={styles.confidenceSection}>
        <Text style={styles.sectionLabel}>Model Confidence</Text>
        <View style={styles.barBackground}>
          <View
            style={[styles.barFill, { width: `${confidencePct}%`, backgroundColor: severityColor }]}
          />
        </View>
        <Text style={styles.confidenceValue}>{confidencePct}%</Text>
      </View>

      {/* Nutrition Advice */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Dietary Recommendation</Text>
        <Text style={styles.cardBody}>{nutrition_advice}</Text>
      </View>

      {/* Food List */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recommended Foods</Text>
        {recommended_foods.map((food, idx) => (
          <View key={idx} style={styles.foodItem}>
            <Text style={styles.bullet}>•</Text>
            <Text style={styles.foodText}>{food}</Text>
          </View>
        ))}
      </View>

      {/* Referral Action */}
      <View style={[styles.card, diagnosis === 'Severe' && styles.urgentCard]}>
        <Text style={styles.cardTitle}>
          {diagnosis === 'Severe' ? '⚠️ Referral Action' : 'Referral Action'}
        </Text>
        <Text style={[styles.cardBody, diagnosis === 'Severe' && styles.urgentText]}>
          {referral_action}
        </Text>
      </View>

      {/* New Screening Button */}
      <TouchableOpacity style={styles.button} onPress={handleNewScreening}>
        <Text style={styles.buttonText}>New Screening</Text>
      </TouchableOpacity>

      {/* Disclaimer */}
      <Text style={styles.disclaimer}>
        ⚕️ This tool is for preliminary screening only. Always consult a qualified healthcare
        professional for diagnosis and treatment.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#1a1a2e',
    padding: 24,
    alignItems: 'center',
  },
  title: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 20,
    textAlign: 'center',
  },
  badge: {
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 50,
    marginBottom: 24,
  },
  badgeText: { color: '#fff', fontSize: 24, fontWeight: '800', textAlign: 'center' },
  confidenceSection: { width: '100%', marginBottom: 20 },
  sectionLabel: { color: '#ccc', fontSize: 13, marginBottom: 6 },
  barBackground: {
    height: 12,
    backgroundColor: '#333',
    borderRadius: 6,
    overflow: 'hidden',
  },
  barFill: { height: 12, borderRadius: 6 },
  confidenceValue: { color: '#fff', fontSize: 13, marginTop: 4, textAlign: 'right' },
  card: {
    width: '100%',
    backgroundColor: '#2d2d44',
    borderRadius: 12,
    padding: 16,
    marginBottom: 14,
  },
  urgentCard: { borderWidth: 2, borderColor: '#e74c3c' },
  cardTitle: { color: '#e74c3c', fontSize: 15, fontWeight: '700', marginBottom: 8 },
  cardBody: { color: '#ddd', fontSize: 14, lineHeight: 21 },
  urgentText: { color: '#ff8080' },
  foodItem: { flexDirection: 'row', alignItems: 'flex-start', marginTop: 4 },
  bullet: { color: '#e74c3c', marginRight: 8, fontSize: 18, lineHeight: 22 },
  foodText: { color: '#ddd', fontSize: 14, flex: 1 },
  button: {
    width: '100%',
    backgroundColor: '#e74c3c',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
  disclaimer: {
    color: '#666',
    fontSize: 11,
    textAlign: 'center',
    lineHeight: 16,
    paddingHorizontal: 8,
  },
});
