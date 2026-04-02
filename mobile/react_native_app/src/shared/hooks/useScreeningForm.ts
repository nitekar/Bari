/**
 * useScreeningForm.ts — Shared screening form logic
 *
 * Extracts form state, validation, and submit handler used by both
 * (tabs)/screening.tsx and (chw)/screening.tsx to eliminate duplication.
 *
 * SECURITY: Input validation with bounds checking on all fields.
 */
import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { useStore } from '../../store/useStore';
import { useAnalyticsStore } from '../../store/analyticsStore';
import {
  predictImage,
  predictMultimodal,
  OfflineError,
} from '../../services/screeningService';

export type ScreeningMode = 'quick' | 'full';

// ── Validation constants ─────────────────────────────────────────────────────
const MAX_NAME_LENGTH = 100;
const MAX_LOCATION_LENGTH = 200;
const AGE_MIN = 0;
const AGE_MAX = 1200; // 100 years in months
const HB_MIN = 0;
const HB_MAX = 25; // g/dL — physiological upper bound

/** Generate a collision-safe unique ID */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Sanitize text input — trim whitespace, enforce max length */
function sanitizeText(text: string, maxLength: number): string {
  return text.trim().slice(0, maxLength);
}

/** Validate numeric input — returns NaN if invalid */
function parseNumericInput(value: string, min: number, max: number): number {
  const num = parseFloat(value);
  if (isNaN(num) || num < min || num > max) return NaN;
  return num;
}

export interface ScreeningFormState {
  mode: ScreeningMode;
  patientName: string;
  patientLocation: string;
  age: string;
  gender: number;
  hbLevel: string;
  offlineQueued: boolean;
}

export interface ScreeningFormActions {
  setMode: (mode: ScreeningMode) => void;
  setPatientName: (v: string) => void;
  setPatientLocation: (v: string) => void;
  setAge: (v: string) => void;
  setGender: (v: number) => void;
  setHbLevel: (v: string) => void;
  handleSubmit: () => Promise<void>;
}

export interface ScreeningFormDerived {
  ageNum: number;
  hbNum: number | null;
  hasAge: boolean;
  hasImage: boolean;
  canSubmit: boolean;
  imageUri: string | null;
  isLoading: boolean;
  error: string | null;
}

export function useScreeningForm() {
  const router = useRouter();

  // ── Local form state ──
  const [mode, setMode] = useState<ScreeningMode>('full');
  const [patientName, setPatientName] = useState('');
  const [patientLocation, setPatientLocation] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState(0);
  const [hbLevel, setHbLevel] = useState('');
  const [offlineQueued, setOfflineQueued] = useState(false);

  // ── Global state ──
  const imageUri = useStore((s) => s.imageUri);
  const userId = useStore((s) => s.userId);
  const isLoading = useStore((s) => s.isLoading);
  const error = useStore((s) => s.error);
  const setResult = useStore((s) => s.setResult);
  const setLoading = useStore((s) => s.setLoading);
  const setError = useStore((s) => s.setError);
  const addToHistory = useStore((s) => s.addToHistory);
  const trackEvent = useAnalyticsStore((s) => s.trackEvent);

  // ── Derived validation ──
  const ageNum = parseNumericInput(age, AGE_MIN, AGE_MAX);
  const hbNum = hbLevel ? parseNumericInput(hbLevel, HB_MIN, HB_MAX) : null;
  const hasAge = !isNaN(ageNum);
  const hasImage = !!imageUri;
  const canSubmit = mode === 'quick' ? hasImage : hasAge && hasImage;

  // ── Sanitized setters ──
  const setSafePatientName = useCallback((v: string) => {
    setPatientName(v.slice(0, MAX_NAME_LENGTH));
  }, []);

  const setSafePatientLocation = useCallback((v: string) => {
    setPatientLocation(v.slice(0, MAX_LOCATION_LENGTH));
  }, []);

  // ── Submit ──
  const handleSubmit = useCallback(async () => {
    if (!hasImage) {
      Alert.alert('Missing Image', 'Please add a conjunctiva image to proceed.');
      return;
    }
    if (mode === 'full' && !hasAge) {
      Alert.alert('Invalid Data', 'Please enter a valid patient age (0–1200 months).');
      return;
    }
    if (mode === 'full' && hbLevel && (hbNum === null || isNaN(hbNum!))) {
      Alert.alert('Invalid Data', `Hemoglobin level must be between ${HB_MIN} and ${HB_MAX} g/dL.`);
      return;
    }

    setOfflineQueued(false);
    setLoading(true);
    setError(null);
    trackEvent('screening_started', { mode });

    // Sanitize text fields before sending
    const safeName = sanitizeText(patientName, MAX_NAME_LENGTH);
    const safeLocation = sanitizeText(patientLocation, MAX_LOCATION_LENGTH);

    try {
      const result =
        mode === 'quick'
          ? await predictImage(imageUri!, userId)
          : await predictMultimodal(
              { imageUri: imageUri!, age: ageNum, gender, hb_level: hbNum },
              userId,
            );

      setResult(result);
      setLoading(false);
      trackEvent('screening_completed', {
        severity: result.prediction,
        confidence: result.confidence,
        mode,
      });
      addToHistory({
        id: generateId(),
        date: new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        mode: mode === 'quick' ? 'image' : 'multimodal',
        age: mode === 'full' ? ageNum : undefined,
        gender: mode === 'full' ? gender : undefined,
        imageUrl: result.imageStoragePath,
        patientName: safeName || undefined,
        patientLocation: safeLocation || undefined,
      });
      router.push('/result');
    } catch (err: any) {
      setLoading(false);
      if (err instanceof OfflineError) {
        setOfflineQueued(true);
        setError(null);
      } else {
        setError(err.message ?? 'Screening failed. Please try again.');
      }
    }
  }, [mode, age, gender, hbLevel, imageUri, patientName, patientLocation, userId]);

  const state: ScreeningFormState = {
    mode,
    patientName,
    patientLocation,
    age,
    gender,
    hbLevel,
    offlineQueued,
  };

  const actions: ScreeningFormActions = {
    setMode,
    setPatientName: setSafePatientName,
    setPatientLocation: setSafePatientLocation,
    setAge,
    setGender,
    setHbLevel,
    handleSubmit,
  };

  const derived: ScreeningFormDerived = {
    ageNum,
    hbNum,
    hasAge,
    hasImage,
    canSubmit,
    imageUri,
    isLoading,
    error,
  };

  return { state, actions, derived };
}
