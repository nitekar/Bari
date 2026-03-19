/**
 * ResultScreen
 * ────────────
 * Step 3 — Displays the full prediction result:
 *   • Diagnosis badge with urgency colour
 *   • Confidence score + per-class probability bars
 *   • Nutritional diet recommendation
 *   • Recommended foods list
 *   • Supplements list
 *   • Referral / medical action card
 *   • Option to start a new screening
 */

import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Image,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';

/* ── Colour palette per severity ──────────────────────────────────────────── */
const SEVERITY_COLORS = {
  'Non-Anemic': {bg: '#e8f5e9', badge: '#2e7d32', bar: '#4caf50'},
  Mild:         {bg: '#fff8e1', badge: '#f57f17', bar: '#ffc107'},
  Moderate:     {bg: '#fce4ec', badge: '#c62828', bar: '#ef5350'},
  Severe:       {bg: '#3e0000', badge: '#b71c1c', bar: '#d32f2f'},
};

const URGENCY_COLORS = {
  Normal:   '#2e7d32',
  Low:      '#f57f17',
  Medium:   '#c62828',
  Critical: '#b71c1c',
};

/* ── Sub-components ──────────────────────────────────────────────────────── */
function ProbabilityBar({label, value, barColor}) {
  return (
    <View style={bar.row}>
      <Text style={bar.label}>{label}</Text>
      <View style={bar.track}>
        <View style={[bar.fill, {width: `${(value * 100).toFixed(0)}%`, backgroundColor: barColor}]} />
      </View>
      <Text style={bar.pct}>{(value * 100).toFixed(1)}%</Text>
    </View>
  );
}

function SectionCard({title, color, children}) {
  return (
    <View style={[card.container, {borderLeftColor: color}]}>
      <Text style={[card.title, {color}]}>{title}</Text>
      {children}
    </View>
  );
}

function BulletList({items, color}) {
  return (
    <View>
      {items.map((item, i) => (
        <View key={i} style={list.row}>
          <Text style={[list.dot, {color}]}>•</Text>
          <Text style={list.text}>{item}</Text>
        </View>
      ))}
    </View>
  );
}

/* ── Main screen ─────────────────────────────────────────────────────────── */
export default function ResultScreen({route, navigation}) {
  const {prediction, imageUri, patientAge, patientGender} = route.params;

  const {
    diagnosis,
    confidence,
    class_probabilities,
    hb_range,
    urgency_level,
    nutrition_advice,
    recommended_foods,
    supplements,
    foods_to_avoid,
    referral_action,
    followup_weeks,
  } = prediction;

  const palette    = SEVERITY_COLORS[diagnosis] || SEVERITY_COLORS['Moderate'];
  const urgColor   = URGENCY_COLORS[urgency_level] || '#555';
  const CLASS_ORDER = ['Non-Anemic', 'Mild', 'Moderate', 'Severe'];
  const confPct    = (confidence * 100).toFixed(1);
  const ageyrs     = (parseInt(patientAge, 10) / 12).toFixed(1);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>

        {/* ── Patient summary strip ────────────────────────────────────── */}
        <View style={styles.patientStrip}>
          <Image source={{uri: imageUri}} style={styles.thumb} resizeMode="cover" />
          <View style={{flex: 1}}>
            <Text style={styles.patientLabel}>Patient</Text>
            <Text style={styles.patientInfo}>
              {patientGender} · {patientAge}m ({ageyrs} yrs)
            </Text>
          </View>
        </View>

        {/* ── Diagnosis badge ──────────────────────────────────────────── */}
        <View style={[styles.diagnosisBadge, {backgroundColor: palette.bg, borderColor: palette.badge}]}>
          <Text style={[styles.diagnosisTitle, {color: palette.badge}]}>{diagnosis}</Text>
          <View style={[styles.urgencyPill, {backgroundColor: urgColor}]}>
            <Text style={styles.urgencyText}>{urgency_level}</Text>
          </View>
          <Text style={styles.confText}>Confidence: {confPct}%</Text>
          <Text style={styles.hbText}>Expected HB: {hb_range}</Text>
        </View>

        {/* ── Confidence bars ──────────────────────────────────────────── */}
        <SectionCard title="Class Probabilities" color="#1a237e">
          {CLASS_ORDER.map(cls => (
            <ProbabilityBar
              key={cls}
              label={cls}
              value={class_probabilities[cls] || 0}
              barColor={SEVERITY_COLORS[cls]?.bar || '#888'}
            />
          ))}
        </SectionCard>

        {/* ── Diet recommendation ──────────────────────────────────────── */}
        <SectionCard title="Dietary Recommendation" color="#2e7d32">
          <Text style={styles.adviceText}>{nutrition_advice}</Text>
        </SectionCard>

        {/* ── Recommended foods ────────────────────────────────────────── */}
        <SectionCard title="Recommended Foods" color="#1565c0">
          <BulletList items={recommended_foods} color="#1565c0" />
        </SectionCard>

        {/* ── Foods to avoid ───────────────────────────────────────────── */}
        {foods_to_avoid?.length > 0 && (
          <SectionCard title="Reduce / Avoid" color="#f57f17">
            <BulletList items={foods_to_avoid} color="#f57f17" />
          </SectionCard>
        )}

        {/* ── Supplements ──────────────────────────────────────────────── */}
        {supplements?.length > 0 && (
          <SectionCard title="Supplements" color="#6a1b9a">
            <BulletList items={supplements} color="#6a1b9a" />
          </SectionCard>
        )}

        {/* ── Referral card ────────────────────────────────────────────── */}
        <View style={[styles.referralCard, {borderColor: urgColor}]}>
          <Text style={[styles.referralTitle, {color: urgColor}]}>
            {urgency_level === 'Critical' ? '🚨  Medical Referral' : '🏥  Medical Guidance'}
          </Text>
          <Text style={styles.referralText}>{referral_action}</Text>
          <Text style={styles.followup}>
            Recommended follow-up: {followup_weeks < 52
              ? `in ${followup_weeks} week${followup_weeks > 1 ? 's' : ''}`
              : 'annually'}
          </Text>
        </View>

        {/* ── Disclaimer ───────────────────────────────────────────────── */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            ⚕️ This result is generated by an AI screening tool and is not a
            medical diagnosis. Always consult a qualified healthcare professional.
          </Text>
        </View>

        {/* ── New screening button ─────────────────────────────────────── */}
        <TouchableOpacity
          style={styles.newBtn}
          onPress={() => navigation.navigate('Capture')}>
          <Text style={styles.newBtnText}>+ New Screening</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

/* ── Styles ──────────────────────────────────────────────────────────────── */
const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f5f5f5'},
  scroll: {padding: 16},

  patientStrip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 16,
    elevation: 2,
    gap: 12,
  },
  thumb: {width: 56, height: 56, borderRadius: 8, borderWidth: 1, borderColor: '#ccc'},
  patientLabel: {fontSize: 11, color: '#888'},
  patientInfo: {fontSize: 15, fontWeight: '600', color: '#222'},

  diagnosisBadge: {
    borderRadius: 12,
    borderWidth: 2,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
    elevation: 3,
  },
  diagnosisTitle: {fontSize: 32, fontWeight: 'bold', marginBottom: 6},
  urgencyPill: {
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 4,
    marginBottom: 8,
  },
  urgencyText: {color: '#fff', fontWeight: 'bold', fontSize: 13},
  confText: {fontSize: 18, fontWeight: '600', color: '#333', marginBottom: 4},
  hbText: {fontSize: 13, color: '#666'},

  adviceText: {fontSize: 14, color: '#333', lineHeight: 22},

  referralCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 2,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
  },
  referralTitle: {fontSize: 16, fontWeight: 'bold', marginBottom: 8},
  referralText: {fontSize: 14, color: '#333', lineHeight: 22, marginBottom: 8},
  followup: {fontSize: 13, color: '#555', fontStyle: 'italic'},

  disclaimer: {
    backgroundColor: '#fff3e0',
    borderRadius: 10,
    padding: 14,
    marginBottom: 16,
  },
  disclaimerText: {fontSize: 12, color: '#e65100', lineHeight: 18},

  newBtn: {
    backgroundColor: '#1a237e',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 8,
  },
  newBtnText: {color: '#fff', fontWeight: 'bold', fontSize: 16},
});

/* ── Probability bar styles ──────────────────────────────────────────────── */
const bar = StyleSheet.create({
  row: {flexDirection: 'row', alignItems: 'center', marginVertical: 4},
  label: {width: 90, fontSize: 12, color: '#555'},
  track: {flex: 1, height: 14, backgroundColor: '#e0e0e0', borderRadius: 7, overflow: 'hidden'},
  fill: {height: '100%', borderRadius: 7},
  pct: {width: 44, fontSize: 12, color: '#333', textAlign: 'right'},
});

/* ── Card styles ─────────────────────────────────────────────────────────── */
const card = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderLeftWidth: 4,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
  },
  title: {fontSize: 15, fontWeight: 'bold', marginBottom: 10},
});

/* ── Bullet list styles ──────────────────────────────────────────────────── */
const list = StyleSheet.create({
  row: {flexDirection: 'row', marginVertical: 2},
  dot: {fontSize: 16, marginRight: 8, lineHeight: 20},
  text: {flex: 1, fontSize: 13, color: '#333', lineHeight: 20},
});
