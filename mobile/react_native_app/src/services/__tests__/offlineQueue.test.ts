/**
 * offlineQueue.test.ts — Unit tests for offline queue pure functions
 */
import {
  createQueuedRequest,
  advanceRetry,
  isReadyToRetry,
  getRetryDelay,
} from '../offlineQueue';

describe('offlineQueue', () => {
  describe('getRetryDelay', () => {
    it('returns 5000ms for retry 0', () => {
      expect(getRetryDelay(0)).toBe(5000);
    });

    it('returns 15000ms for retry 1 (5s * 3)', () => {
      expect(getRetryDelay(1)).toBe(15000);
    });

    it('returns 45000ms for retry 2 (5s * 9)', () => {
      expect(getRetryDelay(2)).toBe(45000);
    });
  });

  describe('createQueuedRequest', () => {
    it('creates a request with correct defaults', () => {
      const req = createQueuedRequest(
        '/predict/multimodal',
        'POST',
        { age: 12, gender: 0 },
        'multipart/form-data',
      );

      expect(req.endpoint).toBe('/predict/multimodal');
      expect(req.method).toBe('POST');
      expect(req.body).toEqual({ age: 12, gender: 0 });
      expect(req.contentType).toBe('multipart/form-data');
      expect(req.retryCount).toBe(0);
      expect(req.maxRetries).toBe(3);
      expect(req.id).toBeTruthy();
      expect(req.createdAt).toBeTruthy();
      expect(req.nextRetryAt).toBeTruthy();
    });

    it('generates unique IDs', () => {
      const req1 = createQueuedRequest('/a', 'GET', null, 'application/json');
      const req2 = createQueuedRequest('/a', 'GET', null, 'application/json');
      expect(req1.id).not.toBe(req2.id);
    });

    it('stores localImageUri when provided', () => {
      const req = createQueuedRequest(
        '/predict/image',
        'POST',
        {},
        'multipart/form-data',
        'file:///tmp/img.jpg',
      );
      expect(req.localImageUri).toBe('file:///tmp/img.jpg');
    });
  });

  describe('advanceRetry', () => {
    it('increments retryCount and updates nextRetryAt', () => {
      const req = createQueuedRequest('/a', 'POST', {}, 'application/json');
      const advanced = advanceRetry(req);

      expect(advanced).not.toBeNull();
      expect(advanced!.retryCount).toBe(1);
      expect(new Date(advanced!.nextRetryAt).getTime()).toBeGreaterThan(Date.now());
    });

    it('returns null when max retries exceeded', () => {
      const req = createQueuedRequest('/a', 'POST', {}, 'application/json');
      const r1 = advanceRetry(req);    // -> retry 1
      const r2 = advanceRetry(r1!);    // -> retry 2
      const r3 = advanceRetry(r2!);    // -> null (max 3 exceeded)

      expect(r3).toBeNull();
    });
  });

  describe('isReadyToRetry', () => {
    it('returns true when nextRetryAt is in the past', () => {
      const req = createQueuedRequest('/a', 'POST', {}, 'application/json');
      req.nextRetryAt = new Date(Date.now() - 1000).toISOString();

      expect(isReadyToRetry(req)).toBe(true);
    });

    it('returns false when nextRetryAt is in the future', () => {
      const req = createQueuedRequest('/a', 'POST', {}, 'application/json');
      req.nextRetryAt = new Date(Date.now() + 60000).toISOString();

      expect(isReadyToRetry(req)).toBe(false);
    });
  });
});
