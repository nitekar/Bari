/**
 * types.ts — TypeScript interfaces for API communication
 */

/** Response from all /predict/* endpoints */
export interface PredictionResponse {
  prediction: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  hb_estimate_gdl?: number | null;
  nutrition: string;
  recommended_foods: string[];
  referral_action: string;

  // New structured decision-support fields (optional — backend may not return these)
  risk_level?: 'low' | 'moderate' | 'high';
  confidence_score?: number;
  recommendations?: {
    diet_plan: string;
    foods_to_include: string[];
    foods_to_avoid: string[];
    urgency_level: 'routine' | 'elevated' | 'urgent';
  };
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
  patient_name?: string | null;
  patient_location?: string | null;
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
