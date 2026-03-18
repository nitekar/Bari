/**
 * CaptureScreen.js
 * ----------------
 * Camera capture UI for conjunctiva image acquisition.
 * Uses Expo Camera to take a photo, previews it, and navigates
 * to PatientFormScreen with the image URI.
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

export default function CaptureScreen({ navigation }) {
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [previewUri, setPreviewUri] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const handleCapture = async () => {
    if (!cameraRef.current || !cameraReady) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
    setPreviewUri(photo.uri);
  };

  const handlePickFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });
    if (!result.canceled) {
      setPreviewUri(result.assets[0].uri);
    }
  };

  const handleConfirm = () => {
    if (!previewUri) {
      Alert.alert('No image', 'Please capture or select an image first.');
      return;
    }
    navigation.navigate('PatientForm', { imageUri: previewUri });
  };

  const handleRetake = () => setPreviewUri(null);

  if (hasPermission === null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#e74c3c" />
        <Text style={styles.infoText}>Requesting camera permission…</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Camera permission denied.</Text>
        <Text style={styles.infoText}>
          Please enable camera access in device settings.
        </Text>
        <TouchableOpacity style={styles.button} onPress={handlePickFromGallery}>
          <Text style={styles.buttonText}>Select from Gallery</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Capture Conjunctiva Image</Text>
      <Text style={styles.subtitle}>
        Gently pull down the lower eyelid and position the inner eyelid in frame.
      </Text>

      {previewUri ? (
        <View style={styles.previewContainer}>
          <Image source={{ uri: previewUri }} style={styles.preview} />
          <View style={styles.row}>
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={handleRetake}>
              <Text style={styles.buttonText}>Retake</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={handleConfirm}>
              <Text style={styles.buttonText}>Use This Image</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.cameraWrapper}>
          <Camera
            ref={cameraRef}
            style={styles.camera}
            type={Camera.Constants.Type.back}
            onCameraReady={() => setCameraReady(true)}
          >
            <View style={styles.overlay}>
              <View style={styles.focusBox} />
            </View>
          </Camera>
          <View style={styles.controls}>
            <TouchableOpacity style={styles.galleryButton} onPress={handlePickFromGallery}>
              <Text style={styles.galleryButtonText}>Gallery</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
              <View style={styles.captureInner} />
            </TouchableOpacity>
            <View style={{ width: 64 }} />
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e', padding: 16 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#1a1a2e', padding: 24 },
  title: { color: '#fff', fontSize: 20, fontWeight: '700', textAlign: 'center', marginBottom: 6 },
  subtitle: { color: '#aaa', fontSize: 13, textAlign: 'center', marginBottom: 16 },
  cameraWrapper: { flex: 1 },
  camera: { flex: 1, borderRadius: 12, overflow: 'hidden' },
  overlay: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  focusBox: {
    width: 200, height: 200,
    borderWidth: 2, borderColor: '#e74c3c',
    borderRadius: 8,
  },
  controls: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 20 },
  captureButton: {
    width: 72, height: 72, borderRadius: 36,
    borderWidth: 4, borderColor: '#e74c3c',
    justifyContent: 'center', alignItems: 'center',
  },
  captureInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#e74c3c' },
  galleryButton: {
    width: 64, height: 40,
    justifyContent: 'center', alignItems: 'center',
    backgroundColor: '#333', borderRadius: 8,
  },
  galleryButtonText: { color: '#fff', fontSize: 13 },
  previewContainer: { flex: 1 },
  preview: { flex: 1, borderRadius: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 12 },
  button: {
    flex: 1, marginHorizontal: 6, padding: 14,
    backgroundColor: '#e74c3c', borderRadius: 10,
    alignItems: 'center',
  },
  secondaryButton: { backgroundColor: '#555' },
  buttonText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  infoText: { color: '#aaa', marginTop: 8, textAlign: 'center' },
  errorText: { color: '#e74c3c', fontSize: 16, fontWeight: '700' },
});
