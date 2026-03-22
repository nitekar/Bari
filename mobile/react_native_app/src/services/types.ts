/**
 * types.ts — TypeScript interfaces for API communication
 */

/** Response from all /predict/* endpoints */
export interface PredictionResponse {
  prediction: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  nutrition: string;
  recommended_foods: string[];
  referral_action: string;
}

/** Request body for /predict/tabular */
export interface TabularRequest {
  age: number;
  gender: number;
  hb_level?: number | null;
}

/** Fields for multimodal request (sent as FormData) */
export interface MultimodalFields {
  imageUri: string;
  age: number;
  gender: number;
  hb_level?: number | null;
}

/** API error shape */
export interface ApiError {
  detail: string;
  code?: number;
}

// ── Supabase Database Row Types ──────────────────────────────────────────────

/** Row in the `screenings` table */
export interface ScreeningRow {
  id: string;
  user_id: string;
  prediction: string;
  confidence: number;
  mode: 'tabular' | 'image' | 'multimodal';
  age?: number;
  gender?: number;
  hb_level?: number | null;
  image_url?: string | null;
  created_at: string;
}

/** Row in the `analytics_events` table */
export interface AnalyticsEventRow {
  id: string;
  user_id: string;
  event_name: string;
  metadata: Record<string, unknown>;
  created_at: string;
}
