/**
 * syncService.ts — Processes the offline queue when connectivity is restored.
 *
 * Call `processSyncQueue()` whenever the device comes back online.
 * Each ready item is executed; on failure it's advanced (exponential backoff)
 * or discarded when max retries are exceeded.
 */
import api from './api';
import { useStore } from '../store/useStore';
import { isReadyToRetry, advanceRetry } from './offlineQueue';
import type { QueuedRequest } from './offlineQueue';

async function executeRequest(req: QueuedRequest): Promise<void> {
  if (req.contentType === 'multipart/form-data' && req.localImageUri) {
    const formData = new FormData();
    formData.append('file', {
      uri: req.localImageUri,
      name: 'conjunctiva.jpg',
      type: 'image/jpeg',
    } as unknown as Blob);

    if (req.body && typeof req.body === 'object') {
      for (const [k, v] of Object.entries(req.body as Record<string, unknown>)) {
        formData.append(k, String(v));
      }
    }

    await api.post(req.endpoint, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  } else {
    const method = req.method.toLowerCase() as 'get' | 'post' | 'put' | 'delete';
    await (api[method] as (url: string, data?: unknown) => Promise<unknown>)(
      req.endpoint,
      req.body,
    );
  }
}

export async function processSyncQueue(): Promise<void> {
  const { offlineQueue, removeFromQueue, addToQueue, setLastSyncedAt } =
    useStore.getState();

  if (offlineQueue.length === 0) return;

  console.log(`[sync] Processing ${offlineQueue.length} queued request(s)`);

  for (const req of [...offlineQueue]) {
    if (!isReadyToRetry(req)) continue;

    try {
      await executeRequest(req);
      removeFromQueue(req.id);
      console.log(`[sync] ✓ ${req.id} synced`);
    } catch (err) {
      const advanced = advanceRetry(req);
      removeFromQueue(req.id);
      if (advanced) {
        addToQueue(advanced);
        console.log(
          `[sync] ✗ ${req.id} failed — retry ${advanced.retryCount}/${advanced.maxRetries}`,
        );
      } else {
        console.warn(`[sync] ✗ ${req.id} max retries exceeded — discarded`);
      }
    }
  }

  setLastSyncedAt(new Date());
}
