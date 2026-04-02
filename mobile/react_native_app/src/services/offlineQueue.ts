/**
 * offlineQueue.ts — Offline request queue with exponential backoff
 *
 * QueuedRequest interface and queue management for offline-first support.
 * Retry schedule: 5s → 15s → 45s (3 retries max, 3x multiplier).
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface QueuedRequest {
  id: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body: unknown;
  contentType: 'application/json' | 'multipart/form-data';
  retryCount: number;
  maxRetries: number;
  nextRetryAt: string; // ISO 8601
  createdAt: string;   // ISO 8601
  /** For image requests — the local URI of the image */
  localImageUri?: string;
}

// ── Constants ────────────────────────────────────────────────────────────────

const BASE_DELAY_MS = 5_000;    // 5 seconds
const BACKOFF_MULTIPLIER = 3;   // 5s → 15s → 45s
const MAX_RETRIES = 3;

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Calculate the next retry delay for a given retry count.
 * retry 0 → 5s, retry 1 → 15s, retry 2 → 45s
 */
export function getRetryDelay(retryCount: number): number {
  return BASE_DELAY_MS * Math.pow(BACKOFF_MULTIPLIER, retryCount);
}

/**
 * Create a new queued request.
 */
export function createQueuedRequest(
  endpoint: string,
  method: QueuedRequest['method'],
  body: unknown,
  contentType: QueuedRequest['contentType'],
  localImageUri?: string,
): QueuedRequest {
  const now = new Date();
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    endpoint,
    method,
    body,
    contentType,
    retryCount: 0,
    maxRetries: MAX_RETRIES,
    nextRetryAt: new Date(now.getTime() + BASE_DELAY_MS).toISOString(),
    createdAt: now.toISOString(),
    localImageUri,
  };
}

/**
 * Advance retry count and compute next retry time.
 * Returns null if max retries exceeded.
 */
export function advanceRetry(request: QueuedRequest): QueuedRequest | null {
  const nextCount = request.retryCount + 1;
  if (nextCount >= request.maxRetries) return null;

  const delay = getRetryDelay(nextCount);
  return {
    ...request,
    retryCount: nextCount,
    nextRetryAt: new Date(Date.now() + delay).toISOString(),
  };
}

/**
 * Check whether a queued request is ready to retry.
 */
export function isReadyToRetry(request: QueuedRequest): boolean {
  return new Date(request.nextRetryAt).getTime() <= Date.now();
}

// ── Feature-to-Offline Behavior Mapping ──────────────────────────────────────
//
// | Feature            | Online              | Offline                              |
// |--------------------|---------------------|--------------------------------------|
// | Tabular screening  | Live API call       | Queued, result shown when synced     |
// | Image screening    | Live upload         | Queued (image stored locally)         |
// | Result view        | Live data           | Cached last result + stale badge     |
// | Analytics          | Local data          | Fully available                      |
// | Referral/PDF       | Generate + share    | Generate locally, share queued       |
