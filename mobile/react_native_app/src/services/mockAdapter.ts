/**
 * mockAdapter.ts — Mock API responses for development without a running backend
 *
 * Activated automatically when __DEV__ is true or BASE_URL is not configured.
 * All responses are delayed by 1.5 seconds to simulate network latency.
 */
import type { AxiosInstance, InternalAxiosRequestConfig, AxiosHeaders } from 'axios';
import { endpoints } from './endpoints';
import type { PredictionResponse } from './types';

const MOCK_DELAY_MS = 1_500;

// ── Mock Responses ──────────────────────────────────────────────────────────

const mockResponses: Record<string, PredictionResponse> = {
  normal: {
    prediction: 'Non-Anemic',
    confidence: 0.92,
    class_probabilities: {
      'Non-Anemic': 0.92,
      Mild: 0.05,
      Moderate: 0.02,
      Severe: 0.01,
    },
    nutrition:
      'Your iron levels appear healthy. Continue eating a balanced diet rich in iron, vitamin B12, and folate to maintain optimal blood health.',
    recommended_foods: [
      'Lean red meat',
      'Spinach and kale',
      'Lentils and beans',
      'Eggs',
      'Fortified cereals',
      'Citrus fruits',
    ],
    referral_action:
      'No immediate medical referral needed. Continue routine check-ups.',
  },
  mild: {
    prediction: 'Mild',
    confidence: 0.78,
    class_probabilities: {
      'Non-Anemic': 0.12,
      Mild: 0.78,
      Moderate: 0.08,
      Severe: 0.02,
    },
    nutrition:
      'You show signs of mild anemia. Increase your intake of iron-rich foods and pair them with vitamin C for better absorption.',
    recommended_foods: [
      'Red meat and liver',
      'Dark leafy greens',
      'Dried fruits (apricots, raisins)',
      'Beans and lentils',
      'Pumpkin seeds',
      'Tofu',
      'Fortified bread',
      'Broccoli',
    ],
    referral_action:
      'Consider scheduling a visit with your healthcare provider for a blood test. Monitor symptoms.',
  },
  moderate: {
    prediction: 'Moderate',
    confidence: 0.84,
    class_probabilities: {
      'Non-Anemic': 0.04,
      Mild: 0.08,
      Moderate: 0.84,
      Severe: 0.04,
    },
    nutrition:
      'Moderate anemia detected. A medical professional should evaluate your iron levels. In the meantime, significantly increase iron-rich food intake.',
    recommended_foods: [
      'Liver and organ meats',
      'Red meat',
      'Shellfish (oysters, clams)',
      'Dark leafy greens (spinach)',
      'Legumes',
      'Pumpkin seeds',
      'Quinoa',
      'Dark chocolate',
      'Fortified cereals',
    ],
    referral_action:
      'Medical referral recommended. Please see a healthcare provider promptly for blood work and possible iron supplementation.',
  },
  severe: {
    prediction: 'Severe',
    confidence: 0.91,
    class_probabilities: {
      'Non-Anemic': 0.01,
      Mild: 0.03,
      Moderate: 0.05,
      Severe: 0.91,
    },
    nutrition:
      'Severe anemia detected. Seek medical attention immediately. Dietary changes alone are insufficient — you likely need medical intervention.',
    recommended_foods: [
      'Liver and organ meats',
      'Red meat',
      'Shellfish',
      'Iron-fortified foods',
      'Dark leafy greens',
      'Dried fruits',
    ],
    referral_action:
      'URGENT: Immediate medical attention required. Severe anemia can be life-threatening. Visit the nearest health facility or call +250 800 22 333.',
  },
};

/**
 * Determine which mock response to return based on the request data.
 */
function selectMockResponse(config: InternalAxiosRequestConfig): PredictionResponse {
  const url = config.url || '';

  // For tabular requests, use age to determine severity
  if (url.includes(endpoints.tabular)) {
    const data = typeof config.data === 'string' ? JSON.parse(config.data) : config.data;
    const age = data?.age ?? 24;
    if (age < 6) return mockResponses.severe;
    if (age < 12) return mockResponses.moderate;
    if (age < 24) return mockResponses.mild;
    return mockResponses.normal;
  }

  // For image/multimodal, rotate through results
  const randomIndex = Math.random();
  if (randomIndex < 0.3) return mockResponses.normal;
  if (randomIndex < 0.55) return mockResponses.mild;
  if (randomIndex < 0.8) return mockResponses.moderate;
  return mockResponses.severe;
}

// ── Attach Mock Adapter ──────────────────────────────────────────────────────

let isMockActive = false;

/**
 * Attach the mock adapter to an Axios instance.
 * All requests to /predict/* will be intercepted and return mock data.
 */
export function attachMockAdapter(axiosInstance: AxiosInstance): void {
  if (isMockActive) return;
  isMockActive = true;

  axiosInstance.interceptors.request.use(async (config) => {
    const url = config.url || '';
    const isPredictEndpoint =
      url.includes('/predict/');

    if (isPredictEndpoint) {
      // Simulate network delay
      await new Promise((resolve) => setTimeout(resolve, MOCK_DELAY_MS));

      const mockData = selectMockResponse(config);

      // Create a mock response by using adapter override
      config.adapter = async () => ({
        data: mockData,
        status: 200,
        statusText: 'OK',
        headers: {} as AxiosHeaders,
        config,
      });
    }

    return config;
  });

  console.log('🧪 Mock API adapter attached — responses will be simulated');
}
