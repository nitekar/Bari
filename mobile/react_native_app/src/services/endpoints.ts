/**
 * endpoints.ts — API endpoint constants with RESTful API contract documentation
 *
 * All endpoints are centralized here for easy maintenance.
 * JSDoc comments describe the RESTful API contract.
 */

export interface ApiEndpoints {
  readonly tabular: string;
  readonly image: string;
  readonly multimodal: string;
  readonly health: string;
}

/**
 * Centralized API endpoints.
 *
 * ── RESTful API Contract ──
 *
 * POST /predict/tabular
 *   Body (JSON):
 *     - age: int (required) — Patient age in months
 *     - gender: int (required) — 0 = Female, 1 = Male
 *     - hb_level: float (optional) — Hemoglobin level in g/dL
 *   Response: PredictionResponse
 *
 * POST /predict/image
 *   Body (multipart/form-data):
 *     - file: File (required) — JPEG conjunctiva image
 *   Response: PredictionResponse
 *
 * POST /predict/multimodal
 *   Body (multipart/form-data):
 *     - file: File (required) — JPEG conjunctiva image
 *     - age: int/string (required) — Patient age in months
 *     - gender: int/string (required) — 0 = Female, 1 = Male
 *     - hb_level: float/string (optional) — Hemoglobin level in g/dL
 *   Response: PredictionResponse
 *
 * GET /health
 *   Response: { "status": "ok" }
 *
 * ── FormData Field Names (multipart/form-data) ──
 * | Field     | Type       | Required | Description                    |
 * |-----------|------------|----------|--------------------------------|
 * | file      | File       | Yes*     | JPEG conjunctiva image         |
 * | age       | int/string | Yes      | Patient age in months          |
 * | gender    | int/string | Yes      | 0 = Female, 1 = Male           |
 * | hb_level  | float/str  | No       | Hemoglobin level in g/dL       |
 *
 * *Required for image and multimodal endpoints only.
 */
export const endpoints: ApiEndpoints = {
  tabular: '/predict/tabular',
  image: '/predict/image',
  multimodal: '/predict/multimodal',
  health: '/health',
} as const;

