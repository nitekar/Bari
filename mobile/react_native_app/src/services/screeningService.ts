/**
 * screeningService.ts — API call functions for all prediction modes
 *
 * After each successful prediction, if a user is authenticated the
 * conjunctiva image (if any) is uploaded to Supabase Storage and
 * the image URL is returned alongside the prediction response.
 */
import * as ImageManipulator from 'expo-image-manipulator';
import api from './api';
import { endpoints } from './endpoints';
import type { PredictionResponse, TabularRequest, MultimodalFields } from './types';
import { uploadConjunctivaImage } from './supabaseStorage';

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Compress image to JPEG at 70% quality and max 800px dimension
 * to reduce upload size and speed up inference.
 */
async function compressImage(uri: string): Promise<string> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: 800 } }],
    { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG },
  );
  return result.uri;
}

/**
 * Build a FormData object from an image URI with optional fields.
 */
function buildImageFormData(
  imageUri: string,
  extraFields?: Record<string, string>,
): FormData {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    name: 'conjunctiva.jpg',
    type: 'image/jpeg',
  } as unknown as Blob);

  if (extraFields) {
    Object.entries(extraFields).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }

  return formData;
}

/**
 * Upload the conjunctiva image to Supabase Storage (if user is authenticated).
 * Returns the storage path or null if skipped / failed.
 */
async function tryUploadImage(
  userId: string | null,
  imageUri: string,
): Promise<string | null> {
  if (!userId) return null;
  try {
    return await uploadConjunctivaImage(userId, imageUri);
  } catch (err) {
    console.warn('Image upload to Supabase skipped:', err);
    return null;
  }
}

// ── Extended result that includes the optional image URL ──────────────────

export interface ScreeningResult extends PredictionResponse {
  /** Supabase Storage path of the uploaded image (if any) */
  imageStoragePath?: string | null;
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Tabular-only prediction (age, gender, optional hb_level).
 * Sends JSON body.
 */
export async function predictTabular(
  data: TabularRequest,
): Promise<ScreeningResult> {
  const body: Record<string, number> = {
    age: data.age,
    gender: data.gender,
  };
  if (data.hb_level != null) {
    body.hb_level = data.hb_level;
  }

  const response = await api.post<PredictionResponse>(endpoints.tabular, body);
  return { ...response.data, imageStoragePath: null };
}

/**
 * Image-only prediction (conjunctiva photo).
 * Sends multipart/form-data.
 * Also uploads the image to Supabase Storage.
 */
export async function predictImage(
  imageUri: string,
  userId?: string | null,
): Promise<ScreeningResult> {
  const compressed = await compressImage(imageUri);

  // Fire ML prediction and Supabase upload in parallel
  const [response, storagePath] = await Promise.all([
    api.post<PredictionResponse>(endpoints.image, buildImageFormData(compressed), {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
    tryUploadImage(userId ?? null, compressed),
  ]);

  return { ...response.data, imageStoragePath: storagePath };
}

/**
 * Multimodal prediction (image + tabular data).
 * Sends multipart/form-data with all fields.
 * Also uploads the image to Supabase Storage.
 */
export async function predictMultimodal(
  data: MultimodalFields,
  userId?: string | null,
): Promise<ScreeningResult> {
  const compressed = await compressImage(data.imageUri);

  const extra: Record<string, string> = {
    age: String(data.age),
    gender: String(data.gender),
  };
  if (data.hb_level != null) {
    extra.hb_level = String(data.hb_level);
  }

  const formData = buildImageFormData(compressed, extra);

  // Fire ML prediction and Supabase upload in parallel
  const [response, storagePath] = await Promise.all([
    api.post<PredictionResponse>(endpoints.multimodal, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
    tryUploadImage(userId ?? null, compressed),
  ]);

  return { ...response.data, imageStoragePath: storagePath };
}
