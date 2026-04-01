/**
 * usePlatformCamera.ts — Cross-platform image picker hook
 *
 * Native (iOS/Android): Uses expo-image-picker
 * Web: Uses <input type="file"> with camera capture
 */
import { useCallback, useRef } from 'react';
import { Platform, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export function usePlatformCamera() {
  // Web file input ref (only used on web)
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const resolveRef = useRef<((uri: string | null) => void) | null>(null);

  const pickImageNative = useCallback(async (): Promise<string | null> => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant photo library access to upload conjunctiva images.',
      );
      return null;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      return result.assets[0].uri;
    }
    return null;
  }, []);

  const takePhotoNative = useCallback(async (): Promise<string | null> => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant camera access to take conjunctiva images.',
      );
      return null;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      return result.assets[0].uri;
    }
    return null;
  }, []);

  const pickImageWeb = useCallback((): Promise<string | null> => {
    return new Promise((resolve) => {
      // Create a hidden file input if it doesn't exist
      if (!fileInputRef.current) {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.setAttribute('capture', 'environment');
        input.style.display = 'none';
        document.body.appendChild(input);

        input.addEventListener('change', () => {
          const file = input.files?.[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = () => {
              resolveRef.current?.(reader.result as string);
              resolveRef.current = null;
            };
            reader.readAsDataURL(file);
          } else {
            resolveRef.current?.(null);
            resolveRef.current = null;
          }
          // Reset input so the same file can be selected again
          input.value = '';
        });

        fileInputRef.current = input;
      }

      resolveRef.current = resolve;
      fileInputRef.current.click();
    });
  }, []);

  const pickImage = useCallback(async (): Promise<string | null> => {
    if (Platform.OS === 'web') {
      return pickImageWeb();
    }
    return pickImageNative();
  }, [pickImageNative, pickImageWeb]);

  const takePhoto = useCallback(async (): Promise<string | null> => {
    if (Platform.OS === 'web') {
      return pickImageWeb(); // Web uses standard file input fallback
    }
    return takePhotoNative();
  }, [takePhotoNative, pickImageWeb]);

  return { pickImage, takePhoto };
}
