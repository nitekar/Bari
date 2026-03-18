/**
 * CaptureScreen
 * Allows the user to capture or select a conjunctiva eye image.
 */

import React, {useState} from 'react';
import {
  Alert,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {RootStackParamList} from '../../App';

type CaptureScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'Capture'
>;

interface Props {
  navigation: CaptureScreenNavigationProp;
}

export default function CaptureScreen({navigation}: Props): React.JSX.Element {
  const [imageUri, setImageUri] = useState<string | null>(null);

  const handleCamera = async () => {
    const result = await launchCamera({
      mediaType: 'photo',
      quality: 0.9,
      cameraType: 'front',
      saveToPhotos: false,
    });

    if (result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri ?? null);
    }
  };

  const handleGallery = async () => {
    const result = await launchImageLibrary({
      mediaType: 'photo',
      quality: 0.9,
      selectionLimit: 1,
    });

    if (result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri ?? null);
    }
  };

  const handleNext = () => {
    if (!imageUri) {
      Alert.alert(
        'No image selected',
        'Please capture or select a conjunctiva image first.',
      );
      return;
    }
    navigation.navigate('PatientInfo', {imageUri});
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Step 1: Capture Eye Image</Text>
      <Text style={styles.subtitle}>
        Take a clear photo of the inner lower eyelid (conjunctiva).
      </Text>

      {imageUri ? (
        <Image source={{uri: imageUri}} style={styles.preview} />
      ) : (
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>👁 No image yet</Text>
        </View>
      )}

      <TouchableOpacity style={styles.buttonPrimary} onPress={handleCamera}>
        <Text style={styles.buttonText}>📷  Open Camera</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.buttonSecondary} onPress={handleGallery}>
        <Text style={styles.buttonTextSecondary}>🖼  Choose from Gallery</Text>
      </TouchableOpacity>

      {imageUri && (
        <TouchableOpacity style={styles.buttonNext} onPress={handleNext}>
          <Text style={styles.buttonText}>Next →</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, alignItems: 'center', padding: 24, backgroundColor: '#f5f7fa'},
  title: {fontSize: 20, fontWeight: 'bold', color: '#1a1a2e', marginBottom: 8},
  subtitle: {fontSize: 14, color: '#555', textAlign: 'center', marginBottom: 20},
  preview: {width: 260, height: 260, borderRadius: 12, marginBottom: 20},
  placeholder: {
    width: 260, height: 260, borderRadius: 12, backgroundColor: '#dde1e7',
    justifyContent: 'center', alignItems: 'center', marginBottom: 20,
  },
  placeholderText: {fontSize: 16, color: '#888'},
  buttonPrimary: {
    backgroundColor: '#1a73e8', paddingVertical: 14, paddingHorizontal: 32,
    borderRadius: 8, marginBottom: 12, width: '80%', alignItems: 'center',
  },
  buttonSecondary: {
    borderWidth: 2, borderColor: '#1a73e8', paddingVertical: 14,
    paddingHorizontal: 32, borderRadius: 8, marginBottom: 12,
    width: '80%', alignItems: 'center',
  },
  buttonNext: {
    backgroundColor: '#34a853', paddingVertical: 14, paddingHorizontal: 32,
    borderRadius: 8, marginTop: 8, width: '80%', alignItems: 'center',
  },
  buttonText: {color: '#fff', fontWeight: 'bold', fontSize: 16},
  buttonTextSecondary: {color: '#1a73e8', fontWeight: 'bold', fontSize: 16},
});
