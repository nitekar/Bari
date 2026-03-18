/**
 * api.js
 * ------
 * Axios-based helper for calling the Anemia Screening inference API.
 *
 * Usage:
 *   import { predictAnemia } from './api';
 *   const result = await predictAnemia(imageUri, age, gender);
 */

import axios from 'axios';

// ---------------------------------------------------------------------------
// API base URL — update to your server's address when deploying
// ---------------------------------------------------------------------------
const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Send a conjunctiva image and patient demographics to the prediction endpoint.
 *
 * @param {string} imageUri    - Local file URI of the captured/selected image
 * @param {number} age         - Patient age in years
 * @param {'Male'|'Female'} gender - Patient gender
 * @returns {Promise<{
 *   diagnosis: string,
 *   confidence: number,
 *   nutrition_advice: string,
 *   recommended_foods: string[],
 *   referral_action: string
 * }>}
 */
export async function predictAnemia(imageUri, age, gender) {
  const formData = new FormData();

  // Append image as a multipart file
  const filename = imageUri.split('/').pop();
  const match = /\.(\w+)$/.exec(filename);
  const mimeType = match ? `image/${match[1].toLowerCase().replace('jpg', 'jpeg')}` : 'image/jpeg';

  formData.append('image', {
    uri: imageUri,
    name: filename || 'conjunctiva.jpg',
    type: mimeType,
  });

  formData.append('age', String(age));
  formData.append('gender', gender);

  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000, // 30 s
  });

  return response.data;
}

/**
 * Check that the inference API is reachable.
 *
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
    return response.data?.status === 'ok';
  } catch {
    return false;
  }
}
