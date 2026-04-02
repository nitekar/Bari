/**
 * screeningService.test.ts — Unit tests for screening service
 */
import { OfflineError } from '../screeningService';

// Mock the api module
jest.mock('../api', () => ({
  __esModule: true,
  default: {
    post: jest.fn(),
  },
}));

// Mock image manipulator
jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn().mockResolvedValue({ uri: 'compressed-uri' }),
  SaveFormat: { JPEG: 'jpeg' },
}));

// Mock supabase storage
jest.mock('../supabaseStorage', () => ({
  uploadConjunctivaImage: jest.fn().mockResolvedValue('user/123.jpg'),
}));

// Mock the store
jest.mock('../../store/useStore', () => ({
  useStore: {
    getState: () => ({
      addToQueue: jest.fn(),
    }),
  },
}));

import api from '../api';
import { predictMultimodal } from '../screeningService';

describe('screeningService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('OfflineError', () => {
    it('has correct name and message', () => {
      const err = new OfflineError();
      expect(err.name).toBe('OfflineError');
      expect(err.message).toContain('offline');
    });

    it('is an instance of Error', () => {
      const err = new OfflineError();
      expect(err).toBeInstanceOf(Error);
    });
  });

  describe('predictMultimodal', () => {
    it('posts to the supported multimodal endpoint and returns a storage path', async () => {
      const mockResponse = {
        data: {
          prediction: 'Moderate',
          confidence: 0.92,
          class_probabilities: { 'Non-Anemic': 0.02, Mild: 0.06, Moderate: 0.92 },
          nutrition: 'Maintain balanced diet',
          recommended_foods: ['Iron-rich foods'],
          referral_action: 'No referral needed',
        },
      };
      (api.post as jest.Mock).mockResolvedValue(mockResponse);

      const result = await predictMultimodal({
        imageUri: 'file:///tmp/image.jpg',
        age: 24,
        gender: 1,
      }, 'user-123');

      expect(api.post).toHaveBeenCalledWith(
        '/predict/multimodal',
        expect.any(FormData),
        expect.objectContaining({
          headers: { 'Content-Type': 'multipart/form-data' },
        }),
      );
      expect(result.prediction).toBe('Moderate');
      expect(result.confidence).toBe(0.92);
      expect(result.imageStoragePath).toBe('user/123.jpg');
    });

    it('includes hb_level when provided', async () => {
      (api.post as jest.Mock).mockResolvedValue({
        data: {
          prediction: 'Mild',
          confidence: 0.8,
          class_probabilities: {},
          nutrition: '',
          recommended_foods: [],
          referral_action: '',
        },
      });

      await predictMultimodal({
        imageUri: 'file:///tmp/image.jpg',
        age: 12,
        gender: 0,
        hb_level: 9.5,
      });

      expect(api.post).toHaveBeenCalledWith(
        '/predict/multimodal',
        expect.any(FormData),
        expect.any(Object),
      );
    });

    it('queues request and throws OfflineError on network failure', async () => {
      (api.post as jest.Mock).mockRejectedValue(new Error('Network Error'));

      await expect(
        predictMultimodal({ imageUri: 'file:///tmp/image.jpg', age: 24, gender: 1 }),
      ).rejects.toThrow(OfflineError);
    });

    it('re-throws non-network errors', async () => {
      (api.post as jest.Mock).mockRejectedValue(new Error('Server error: 500'));

      await expect(
        predictMultimodal({ imageUri: 'file:///tmp/image.jpg', age: 24, gender: 1 }),
      ).rejects.toThrow('Server error: 500');
    });
  });
});
