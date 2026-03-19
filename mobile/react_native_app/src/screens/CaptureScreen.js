/**
 * CaptureScreen
 * ─────────────
 * Step 1 of the anemia screening workflow.
 * Allows the user to:
 *   • Take a new photo with the device camera
 *   • Select an existing image from the gallery
 *
 * On confirmation the captured image URI is passed to PatientFormScreen.
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';

const IMAGE_OPTIONS = {
  mediaType: 'photo',
  quality: 0.9,
  maxWidth: 1024,
  maxHeight: 1024,
  includeBase64: false,
};

export default function CaptureScreen({navigation}) {
  const [imageUri, setImageUri] = useState(null);
  const [loading, setLoading] = useState(false);

  /* ── Image pickers ─────────────────────────────────────────────────────── */
  const openCamera = () => {
    setLoading(true);
    launchCamera(IMAGE_OPTIONS, response => {
      setLoading(false);
      if (response.didCancel) return;
      if (response.errorCode) {
        Alert.alert('Camera Error', response.errorMessage || 'Unknown error');
        return;
      }
      if (response.assets && response.assets.length > 0) {
        setImageUri(response.assets[0].uri);
      }
    });
  };

  const openGallery = () => {
    setLoading(true);
    launchImageLibrary(IMAGE_OPTIONS, response => {
      setLoading(false);
      if (response.didCancel) return;
      if (response.errorCode) {
        Alert.alert('Gallery Error', response.errorMessage || 'Unknown error');
        return;
      }
      if (response.assets && response.assets.length > 0) {
        setImageUri(response.assets[0].uri);
      }
    });
  };

  const handleNext = () => {
    if (!imageUri) {
      Alert.alert('No Image', 'Please capture or select a conjunctiva image first.');
      return;
    }
    navigation.navigate('PatientForm', {imageUri});
  };

  /* ── Render ────────────────────────────────────────────────────────────── */
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Conjunctiva Image</Text>
          <Text style={styles.subtitle}>
            Position the lower eyelid so the conjunctiva (inner pink lining) is
            clearly visible, then capture or select the image.
          </Text>
        </View>

        {/* Image preview */}
        <View style={styles.previewContainer}>
          {imageUri ? (
            <Image source={{uri: imageUri}} style={styles.preview} resizeMode="cover" />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderIcon}>👁️</Text>
              <Text style={styles.placeholderText}>No image selected</Text>
            </View>
          )}
        </View>

        {/* Guide tips */}
        <View style={styles.tipsBox}>
          <Text style={styles.tipsTitle}>Capture Tips</Text>
          {[
            'Pull down the lower eyelid gently',
            'Ensure good natural or ambient lighting',
            'Hold camera 10–15 cm from the eye',
            'Keep image in focus — avoid motion blur',
          ].map((tip, i) => (
            <Text key={i} style={styles.tipItem}>
              • {tip}
            </Text>
          ))}
        </View>

        {/* Buttons */}
        {loading ? (
          <ActivityIndicator size="large" color="#1a237e" style={{marginTop: 24}} />
        ) : (
          <View style={styles.buttonRow}>
            <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={openGallery}>
              <Text style={styles.btnTextSecondary}>Gallery</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.btn, styles.btnPrimary]} onPress={openCamera}>
              <Text style={styles.btnTextPrimary}>Camera</Text>
            </TouchableOpacity>
          </View>
        )}

        {imageUri && (
          <TouchableOpacity style={[styles.btn, styles.btnNext]} onPress={handleNext}>
            <Text style={styles.btnTextPrimary}>Next → Patient Info</Text>
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

  header: {alignItems: 'center', marginBottom: 20},
  title: {fontSize: 22, fontWeight: 'bold', color: '#1a237e'},
  subtitle: {fontSize: 14, color: '#555', textAlign: 'center', marginTop: 6, lineHeight: 20},

  previewContainer: {
    width: 280,
    height: 280,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#ddd',
    marginBottom: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 6,
  },
  preview: {width: '100%', height: '100%'},
  placeholder: {flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#e8eaf6'},
  placeholderIcon: {fontSize: 56},
  placeholderText: {fontSize: 14, color: '#888', marginTop: 8},

  tipsBox: {
    backgroundColor: '#e8f5e9',
    borderRadius: 10,
    padding: 14,
    width: '100%',
    marginBottom: 24,
  },
  tipsTitle: {fontWeight: 'bold', fontSize: 14, color: '#2e7d32', marginBottom: 6},
  tipItem: {fontSize: 13, color: '#444', marginBottom: 3},

  buttonRow: {flexDirection: 'row', gap: 12, marginBottom: 12},
  btn: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 10,
    alignItems: 'center',
    minWidth: 130,
  },
  btnPrimary: {backgroundColor: '#1a237e'},
  btnSecondary: {backgroundColor: '#fff', borderWidth: 2, borderColor: '#1a237e'},
  btnNext: {backgroundColor: '#2e7d32', width: '100%', marginTop: 4},
  btnTextPrimary: {color: '#fff', fontWeight: 'bold', fontSize: 15},
  btnTextSecondary: {color: '#1a237e', fontWeight: 'bold', fontSize: 15},
});
