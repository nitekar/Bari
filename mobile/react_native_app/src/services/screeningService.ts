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
import type { PredictionResponse, MultimodalFields } from './types';
import { uploadConjunctivaImage } from './supabaseStorage';
import { createQueuedRequest } from './offlineQueue';
import { useStore } from '../store/useStore';

/** Thrown when the device is offline and the request has been queued. */
export class OfflineError extends Error {
  constructor() {
    super('You are offline. The request has been queued and will sync when you reconnect.');
    this.name = 'OfflineError';
  }
}

function isNetworkError(err: unknown): boolean {
  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    return (
      msg.includes('network') ||
      msg.includes('connection') ||
      msg.includes('reach the server') ||
      msg.includes('timeout')
    );
  }
  return false;
}

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
 * Image-only prediction (conjunctiva photo).
 * Sends multipart/form-data.
 * Also uploads the image to Supabase Storage.
 */
export async function predictImage(
  imageUri: string,
  userId?: string | null,
): Promise<ScreeningResult> {
  const compressed = await compressImage(imageUri);

  try {
    const [response, storagePath] = await Promise.all([
      api.post<PredictionResponse>(endpoints.image, buildImageFormData(compressed), {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
      tryUploadImage(userId ?? null, compressed),
    ]);
    return { ...response.data, imageStoragePath: storagePath };
  } catch (err) {
    if (isNetworkError(err)) {
      useStore.getState().addToQueue(
        createQueuedRequest(endpoints.image, 'POST', {}, 'multipart/form-data', compressed),
      );
      throw new OfflineError();
    }
    throw err;
  }
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

  try {
    const [response, storagePath] = await Promise.all([
      api.post<PredictionResponse>(endpoints.multimodal, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
      tryUploadImage(userId ?? null, compressed),
    ]);
    return { ...response.data, imageStoragePath: storagePath };
  } catch (err) {
    if (isNetworkError(err)) {
      useStore.getState().addToQueue(
        createQueuedRequest(endpoints.multimodal, 'POST', extra, 'multipart/form-data', compressed),
      );
      throw new OfflineError();
    }
    throw err;
  }
}
