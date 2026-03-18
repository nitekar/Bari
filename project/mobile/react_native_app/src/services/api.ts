/**
 * API Service
 * Handles all communication with the Bari Anemia Screening REST API.
 */

import axios from 'axios';
import {PredictionResult} from '../../App';

// Update this URL to match your deployed API endpoint.
const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 s — model inference can take a moment
  headers: {'Content-Type': 'multipart/form-data'},
});

/**
 * Send a conjunctiva image and patient metadata to the /predict endpoint.
 *
 * @param imageUri  - Local file URI of the captured eye image.
 * @param age       - Patient age in months.
 * @param gender    - "Male" or "Female".
 * @returns         Prediction result from the API.
 */
export async function predictAnemia(
  imageUri: string,
  age: number,
  gender: string,
): Promise<PredictionResult> {
  const formData = new FormData();

  // Attach the image file
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'conjunctiva.jpg',
  } as any);

  formData.append('age', String(age));
  formData.append('gender', gender);

  const response = await api.post<PredictionResult>('/predict', formData);
  return response.data;
}

/**
 * Health check — verify the API is reachable.
 */
export async function healthCheck(): Promise<boolean> {
  try {
    await api.get('/health');
    return true;
  } catch {
    return false;
  }
}
