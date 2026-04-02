# Bari — Anemia Screening System

Bari is a cross-platform anemia screening application that combines a multimodal machine learning backend with a mobile-first frontend. It allows community health workers, parents, and clinicians to screen for anemia severity using a conjunctiva (inner eyelid) photograph and/or patient clinical data, then delivers severity classification, nutritional guidance, and referral recommendations — all without laboratory equipment.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Use Cases and Implementation](#use-cases-and-implementation)
5. [Input / Output Reference](#input--output-reference)
6. [Running the Application](#running-the-application)
   - [Backend (FastAPI)](#1-backend-fastapi)
   - [Mobile App (Expo Go)](#2-mobile-app-expo-go)
   - [Web App (Vercel)](#3-web-app-vercel)
   - [Android APK](#4-android-apk)
   - [Docker](#5-docker)
7. [API Endpoints](#api-endpoints)
8. [Example Requests](#example-requests)
9. [Database Schema](#database-schema)
10. [Environment Variables](#environment-variables)
11. [Running Tests](#running-tests)
12. [Generating Model Files](#generating-model-files)
13. [Dataset Citation](#dataset-citation)
14. [Clinical Disclaimer](#clinical-disclaimer)

---

## System Overview

Bari classifies anemia severity into four WHO-aligned categories:

| Class | Hemoglobin — Female (g/dL) | Hemoglobin — Male (g/dL) |
|-------|---------------------------|--------------------------|
| **Non-Anemic** | ≥ 12.0 | ≥ 13.5 |
| **Mild** | 10.0 – 11.9 | 11.0 – 13.4 |
| **Moderate** | 7.0 – 9.9 | 7.0 – 9.9 |
| **Severe** | < 7.0 | < 7.0 |

Three prediction modes are supported:

| Mode | Input | Models used |
|------|-------|-------------|
| **Image only** | Conjunctiva photo | Visual TFLite (MobileNetV2) |
| **Multimodal with HB** | Photo + age + gender + hemoglobin | Fusion TFLite + RF (3-feature) |
| **Multimodal without HB** | Photo + age + gender | Fusion TFLite + RF (2-feature) |

All modes use a **3-tier fallback** strategy to guarantee a response even when models are partially unavailable:
- **Tier 1** — TFLite fusion (best accuracy)
- **Tier 2** — Weighted average of visual (70%) and tabular (30%) predictions
- **Tier 3** — Best individual model (visual or RF alone)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                            │
│  React Native / Expo (iOS, Android, Web)                    │
│  Role groups: (admin) · (chw) · (parent)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS · API_KEY header
┌────────────────────────▼────────────────────────────────────┐
│                   BACKEND LAYER (Railway)                   │
│  FastAPI · uvicorn · slowapi (rate limiting)                │
│                                                             │
│  POST /predict/image                                        │
│  POST /predict/multimodal                                   │
│  GET  /health                                               │
└──────┬─────────────────────────────────────────┬────────────┘
       │                                         │
┌──────▼──────────────┐               ┌──────────▼──────────┐
│   ML INFERENCE      │               │   SUPABASE          │
│                     │               │                     │
│ Visual TFLite       │               │ auth.users          │
│ MobileNetV2         │               │ public.profiles     │
│ [1,160,160,3]→probs │               │ public.screenings   │
│                     │               │ public.children     │
│ Tabular RF + Scaler │               │ public.sleep_logs   │
│ [HB,Age,Gender]     │               │ public.feeding_logs │
│ [Age,Gender]        │               │ storage: conjunctiva│
│                     │               │         -images     │
│ Fusion TFLite       │               └─────────────────────┘
│ image+tab → class   │
└─────────────────────┘
```

---

## Directory Structure

```
Bari/
├── README.md
│
├── app/                            # FastAPI application
│   ├── main.py                     # Entry point, endpoints, auth middleware
│   ├── schemas/
│   │   ├── request.py              # Pydantic input models
│   │   └── response.py             # PredictionResponse, HealthResponse
│   ├── services/
│   │   ├── inference.py            # TFLite + RF inference, fusion, fallback
│   │   ├── nutrition.py            # Guidance generation from prediction
│   │   └── preprocessing.py        # Image resize/normalize, tabular scaling
│   └── utils/
│       └── image_utils.py          # MIME type validation
│
├── api/
│   ├── routes.py                   # /health endpoint
│   └── __init__.py
│
├── utils/
│   └── nutrition.py                # Structured recommendation builder
│
├── models/
│   └── saved_models/
│       ├── visual_model.tflite     # MobileNetV2 image-only model
│       ├── multimodal_model.tflite # Fusion model (image + HB + age + gender)
│       └── multimodal_no_hb_model.tflite  # Fusion model (image + age + gender)
│
├── Notebook/
│   ├── Bari.ipynb                  # Full training pipeline
│   └── models/
│       ├── tabular_with_hb.pkl     # RF + StandardScaler (3 features)
│       └── tabular_no_hb.pkl       # RF + StandardScaler (2 features)
│
├── tests/                          # Backend unit tests (35 tests)
│   ├── test_inference.py
│   ├── test_api.py
│   ├── test_api_mocked.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_split_manifest.py
│   ├── test_dataset_integrity.py
│   └── test_feature_schema.py
│
├── mobile/
│   └── react_native_app/
│       ├── app/                    # Expo Router screens
│       │   ├── _layout.tsx         # Root layout, auth guard, role routing
│       │   ├── onboarding.tsx      # First-time welcome slides
│       │   ├── eula-welcome.tsx    # EULA / consent screen
│       │   ├── auth.tsx            # Sign in / sign up / password reset
│       │   ├── result.tsx          # Screening result display
│       │   ├── image-capture.tsx   # Camera / gallery capture
│       │   ├── education.tsx       # Education hub (guest + authenticated)
│       │   ├── settings.tsx        # Language, preferences
│       │   ├── (admin)/            # Admin role screens
│       │   │   ├── index.tsx       # Admin dashboard
│       │   │   ├── users.tsx       # All registered users
│       │   │   ├── analytics.tsx   # Severity distribution charts
│       │   │   └── audit.tsx       # Full screening audit log
│       │   ├── (chw)/              # Community Health Worker screens
│       │   │   ├── index.tsx       # CHW dashboard
│       │   │   ├── screening.tsx   # Run a screening
│       │   │   ├── patients.tsx    # Patient list
│       │   │   ├── followup.tsx    # Follow-up scheduling
│       │   │   └── profile.tsx     # CHW profile
│       │   └── (parent)/           # Parent / caregiver screens
│       │       ├── index.tsx       # Home dashboard
│       │       ├── baby.tsx        # Child profile & milestones
│       │       ├── results.tsx     # Screening history
│       │       ├── education.tsx   # Education tab
│       │       └── profile.tsx     # Parent profile
│       ├── src/
│       │   ├── services/
│       │   │   ├── supabase.ts     # Supabase client init
│       │   │   ├── supabaseAuth.ts # Sign in/up, role management, email redirect
│       │   │   ├── supabaseDb.ts   # Screenings, sleep, feeding, analytics DB helpers
│       │   │   └── api.ts          # Axios client for FastAPI backend
│       │   ├── store/
│       │   │   └── useStore.ts     # Zustand store (history, child profile, offline queue)
│       │   ├── i18n/
│       │   │   └── translations.ts # EN / FR / RW translations
│       │   └── data/               # Static education content
│       ├── app.config.js           # Expo dynamic config (reads .env)
│       ├── eas.json                # EAS build profiles
│       ├── vercel.json             # Vercel web deployment config
│       └── supabase_migration.sql  # Full DB schema + RLS policies
│
├── Dockerfile                      # Production container
├── requirements.txt                # Python dependencies
└── pytest.ini                      # Test configuration
```

---

## Use Cases and Implementation

This section maps each use case to the screens and backend logic that implement it.

### UC-01 · Register and Sign In

**Actor:** Parent, CHW, Admin  
**Screens:** `auth.tsx`  
**Implementation:**
- User selects role (Parent or CHW) during sign-up
- `supabaseAuth.ts → signUpWithRole()` calls `supabase.auth.signUp()` with `raw_user_meta_data: { role, full_name }`
- A Supabase trigger (`handle_new_user`) automatically creates a row in `public.profiles` with the claimed role; admin self-registration is blocked by the trigger
- On sign-in, `supabaseAuth.ts → signIn()` authenticates and `_layout.tsx` reads the profile role to route the user to `/(admin)`, `/(chw)`, or `/(parent)`
- "Remember me" persists credentials to `expo-secure-store` for automatic pre-fill

**Input → Output:**
| Input | Output |
|-------|--------|
| Email, password, full name, role | Supabase session token + profile row created; user routed to role dashboard |

---

### UC-02 · Screen for Anemia (Image Only)

**Actor:** CHW, Parent  
**Screens:** `image-capture.tsx` → `(chw)/screening.tsx` → `result.tsx`  
**Backend:** `POST /predict/image`  
**Implementation:**
- User photographs the patient's inner eyelid using `expo-image-picker` (camera or gallery)
- Image is converted to base64 / FormData and sent to the FastAPI backend with `API_KEY` header
- Backend: `validate_image_content_type()` → `preprocess_image_bytes()` (resize to 160×160, normalize to [0,1]) → `predict_visual()` (TFLite inference) → `build_visual_probabilities_dict()` → `get_full_guidance()` → `PredictionResponse`
- Result screen displays severity badge, confidence bar, recommended foods, and referral action
- `addToHistory()` in the Zustand store saves the record locally and persists it to `public.screenings` via `saveScreeningResult()`

**Input → Output:**
| Input | Output |
|-------|--------|
| JPEG/PNG conjunctiva image (any resolution) | Severity class, confidence, per-class probabilities, dietary advice, referral action |

---

### UC-03 · Screen for Anemia (Multimodal)

**Actor:** CHW  
**Screens:** `(chw)/screening.tsx` → `result.tsx`  
**Backend:** `POST /predict/multimodal`  
**Implementation:**
- CHW captures image and enters age (months), gender, and optionally hemoglobin level
- Backend routes to the correct fusion model:
  - With HB → `multimodal_model.tflite` + `tabular_with_hb.pkl` (3-feature RF)
  - Without HB → `multimodal_no_hb_model.tflite` + `tabular_no_hb.pkl` (2-feature RF)
- 3-tier fallback: TFLite fusion → weighted average (visual 70% / tabular 30%) → best individual model
- `fusion_strategy` field in response indicates which tier was used
- Result is saved to `public.screenings` with `patient_name` and `patient_location`

**Input → Output:**
| Input | Output |
|-------|--------|
| Image + age (months) + gender (0/1) + HB level (optional, g/dL) | Severity class, confidence, HB estimate, structured recommendations, referral action, fusion strategy used |

---

### UC-04 · View Screening History

**Actor:** CHW, Parent  
**Screens:** `(parent)/results.tsx`, `(chw)/patients.tsx`  
**Implementation:**
- On login, `loadHistoryFromSupabase(userId)` fetches the user's records from `public.screenings` (scoped by RLS: `auth.uid() = user_id`)
- Results are stored in the Zustand store's `history` array
- Parent sees a timeline of their child's screenings with severity badges and dates
- CHW sees a patient list with most recent screening outcome per patient

**Input → Output:**
| Input | Output |
|-------|--------|
| Authenticated user session | List of past screenings with prediction, confidence, date, patient name, mode |

---

### UC-05 · View Nutritional Education

**Actor:** Parent, Guest  
**Screens:** `education.tsx`, `edu-nutrition.tsx`, `edu-feeding.tsx`, `edu-milestones.tsx`, `edu-activities.tsx`, `edu-feedingplan.tsx`, `edu-anemia.tsx`  
**Implementation:**
- Unauthenticated guests see a curated info view with an explanation of Bari and a call-to-action to register
- Authenticated parents access the full education hub: feeding plans, growth milestones, activity checklists, anemia nutrition guides
- Content is loaded from static TypeScript data files in `src/data/` (no network call required)
- Progress is tracked in the Zustand store via `completedItems` and displayed as an overall completion percentage

**Input → Output:**
| Input | Output |
|-------|--------|
| Tap on education category | Detailed guide content, checklist items, feeding stage recommendations |

---

### UC-06 · Track Child Health (Baby Profile)

**Actor:** Parent  
**Screens:** `(parent)/baby.tsx`, `parent-sleep.tsx`, `parent-feeding.tsx`, `parent-development.tsx`  
**Implementation:**
- Parent sets child's name, age (months), and gender in `setChildProfile()` (Zustand store)
- Sleep logs submitted via `parent-sleep.tsx` → `saveSleepLog()` → `public.sleep_logs`
- Feeding logs submitted via `parent-feeding.tsx` → `saveFeedingLog()` → `public.feeding_logs`
- Development milestones tracked locally via `completedItems` in the store
- All data scoped to the authenticated user by RLS policies

**Input → Output:**
| Input | Output |
|-------|--------|
| Sleep start/end times, feeding type/quantity | Persisted logs with date; visible in parent dashboard history |

---

### UC-07 · Admin — View All Users

**Actor:** Admin  
**Screens:** `(admin)/users.tsx`  
**Implementation:**
- Queries `public.profiles` with `.select('id, role, full_name')`
- RLS: `public.is_admin()` security definer function checks caller's role without triggering recursion
- Displays all registered users with their role badge and user ID

**Input → Output:**
| Input | Output |
|-------|--------|
| Admin session | List of all users with role (admin / chw / parent) and full name |

---

### UC-08 · Admin — Analytics and Audit Log

**Actor:** Admin  
**Screens:** `(admin)/analytics.tsx`, `(admin)/audit.tsx`  
**Implementation:**
- Both screens call `getAllScreenings()` → `supabase.from('screenings').select('*')` without user filter
- RLS policy "Admins view all screenings" allows this when `is_admin()` returns true
- `analytics.tsx` computes severity distribution bars and screening mode breakdown from the full dataset
- `audit.tsx` shows a filterable log (All / Severe / Moderate) with patient name, location, confidence, and record ID

**Input → Output:**
| Input | Output |
|-------|--------|
| Admin session | Severity distribution chart, mode breakdown counts, full audit log with filters |

---

### UC-09 · Consent and Onboarding

**Actor:** All users (first launch)  
**Screens:** `onboarding.tsx`, `eula-welcome.tsx`  
**Implementation:**
- First-time users see a 3-slide horizontal onboarding scroll (`hasSeenOnboarding` flag in Zustand store)
- All users must accept the EULA before proceeding (`hasAcceptedEula` flag)
- `_layout.tsx` intercepts navigation: if either flag is false, the user is redirected regardless of auth state
- EULA text is fully internationalised (EN / FR / RW) via `translations.ts`
- Declining the EULA redirects to the auth screen

---

### UC-10 · Language Selection

**Actor:** All users  
**Screens:** `settings.tsx`, language switcher in headers  
**Implementation:**
- `useStore → language` holds the current locale (`en`, `fr`, `rw`)
- `useTranslation()` hook returns the matching translation object from `translations.ts`
- All UI strings (labels, titles, EULA, education content) update immediately without restart
- Selection persisted via Zustand `persist` middleware (localStorage on web, SecureStore on native)

---

### UC-11 · Offline Support and Sync

**Actor:** CHW (field use without connectivity)  
**Implementation:**
- `useNetworkStatus()` hook monitors connectivity (`navigator.onLine` on web, `expo-network` on native)
- Failed API requests are queued in `offlineQueue` (Zustand store) with exponential backoff metadata
- `processSyncQueue()` is called when connectivity is restored, replaying queued requests
- `OfflineIndicator` banner shown when device is offline with last-synced timestamp and pending queue count

---

## Input / Output Reference

### POST `/predict/image` — Image Prediction

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Conjunctiva photo — JPEG or PNG, any resolution |

**Preprocessing pipeline:**
1. MIME type validated (`image/jpeg` or `image/png`)
2. Decoded with Pillow → converted to RGB
3. Resized to 160 × 160 pixels (bilinear)
4. Normalized: pixel values divided by 255.0
5. Shape expanded to `[1, 160, 160, 3]` float32

---

### POST `/predict/multimodal` — Multimodal Fusion Prediction

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Conjunctiva image (JPEG / PNG) |
| `age` | float | Yes | Patient age in **months** (0 – 1200) |
| `gender` | int | Yes | `0` = Male, `1` = Female |
| `hb_level` | float | No | Hemoglobin in g/dL (0 – 25). Omit when unavailable |

**Tabular preprocessing:**
- With HB → feature vector `[HB_LEVEL, Age(months), GENDER]` → `StandardScaler` (from `tabular_with_hb.pkl`)
- Without HB → feature vector `[Age(months), GENDER]` → `StandardScaler` (from `tabular_no_hb.pkl`)

---

### Shared Response Schema (both endpoints)

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"Non-Anemic"` / `"Mild"` / `"Moderate"` / `"Severe"` |
| `confidence` | float | Top-class probability (0.0 – 1.0) |
| `confidence_score` | float | Alias of `confidence` for frontend use |
| `class_probabilities` | object | `{ "Non-Anemic": f, "Mild": f, "Moderate": f, "Severe": f }` |
| `hb_estimate_gdl` | float \| null | Estimated hemoglobin level (g/dL) derived from class |
| `risk_level` | string | `"low"` / `"moderate"` / `"high"` |
| `nutrition` | string | Short dietary advice paragraph |
| `recommended_foods` | list[string] | Specific iron-rich food suggestions |
| `referral_action` | string | Clinical follow-up recommendation |
| `recommendations` | object | Structured: `diet_plan`, `foods_to_include`, `foods_to_avoid`, `urgency_level` |
| `fusion_strategy` | string | Multimodal only: `"tflite_fusion"` / `"weighted_average"` / `"best_individual"` |

---

### GET `/health`

| Field | Description |
|-------|-------------|
| `status` | `"ok"` if all critical models loaded, `"degraded"` otherwise |
| `version` | API version string |
| `models_loaded` | Object — `true`/`false` per model (visual, severity_classifier, etc.) |

---

### Feature Vector Summary

| Endpoint | Variant | Feature vector |
|----------|---------|----------------|
| `/predict/image` | — | `[1, 160, 160, 3]` float32 image array |
| `/predict/multimodal` | with HB | Image `[1,160,160,3]` + tabular `[1,3]`: `[HB, Age, Gender]` scaled |
| `/predict/multimodal` | no HB | Image `[1,160,160,3]` + tabular `[1,2]`: `[Age, Gender]` scaled |

---

## Running the Application

### 1. Backend (FastAPI)

**Prerequisites:** Python 3.11, pip

```bash
# Clone the repo and navigate to project root
cd Bari

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export API_KEY=your-api-key
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_ANON_KEY=your-anon-key

# Run (development)
uvicorn app.main:app --reload --port 8000

# Run (production)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

**Production URL:** https://web-production-c7c1.up.railway.app

---

### 2. Mobile App (Expo Go)

**Prerequisites:** Node.js 18+, Expo Go app installed on your phone

```bash
cd mobile/react_native_app

# Install dependencies
npm install

# Create .env file with credentials
cp .env.example .env
# Edit .env with your Supabase and API values

# Start development server (tunnel mode for physical device)
npx expo start --tunnel
```

Scan the QR code shown in the terminal with the **Expo Go** app (Android or iOS).

For web preview in browser:
```bash
npx expo start --web
```

**.env file format:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
API_BASE_URL=https://your-railway-url.up.railway.app
API_KEY=your-api-key
```

---

### 3. Web App (Vercel)

**Live URL:** Deployed automatically on push to `main` via Vercel.

To run locally:
```bash
cd mobile/react_native_app
npm install
npx expo export --platform web  # builds to dist/
npx serve dist                  # serve locally
```

**Vercel configuration** (`vercel.json`):
- Build command: `npx expo export --platform web`
- Output directory: `dist`
- SPA rewrite: all routes → `index.html`

Required Vercel environment variables (set in Vercel dashboard → Settings → Environment Variables):
```
SUPABASE_URL
SUPABASE_ANON_KEY
API_BASE_URL
API_KEY
```

---

### 4. Android APK

The APK is built via EAS Build (Expo Application Services) in the cloud.

```bash
cd mobile/react_native_app

# Install EAS CLI
npm install -g eas-cli

# Log in to Expo account
npx eas login

# Build preview APK (internal distribution)
npx eas build --profile preview --platform android

# Build production AAB (for Play Store)
npx eas build --profile production --platform android
```

Track builds at: https://expo.dev/accounts/niteka/projects/anemia-screening/builds

EAS build profiles (`eas.json`):

| Profile | Android output | Distribution |
|---------|---------------|--------------|
| `development` | APK | Internal (dev client) |
| `preview` | APK | Internal (shareable) |
| `production` | AAB | Play Store |

---

### 5. Docker

```bash
# Build image
docker build -t bari-api .

# Run container
docker run -p 8000:8000 \
  -e API_KEY=your-key \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_ANON_KEY=your-anon-key \
  bari-api

# Or with Docker Compose (if available)
docker compose up --build
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | None | System health and model status |
| GET | `/` | None | Root — redirects to health |
| POST | `/predict/image` | `X-API-Key` | Image-only anemia prediction |
| POST | `/predict/multimodal` | `X-API-Key` | Fusion prediction (image + clinical) |

**Authentication:** All prediction endpoints require the `X-API-Key` header matching the `API_KEY` environment variable.

**Rate limiting:** 60 requests per minute per IP (via slowapi).

---

## Example Requests

### Image prediction

```bash
curl -X POST https://web-production-c7c1.up.railway.app/predict/image \
  -H "X-API-Key: your-api-key" \
  -F "file=@conjunctiva.jpg"
```

### Multimodal prediction (with HB)

```bash
curl -X POST https://web-production-c7c1.up.railway.app/predict/multimodal \
  -H "X-API-Key: your-api-key" \
  -F "file=@conjunctiva.jpg" \
  -F "age=312" \
  -F "gender=1" \
  -F "hb_level=9.5"
```

### Python client

```python
import requests

url = "https://web-production-c7c1.up.railway.app/predict/multimodal"
headers = {"X-API-Key": "your-api-key"}

with open("conjunctiva.jpg", "rb") as img:
    resp = requests.post(url, headers=headers,
        files={"file": img},
        data={"age": 312, "gender": 1, "hb_level": 9.5}
    )

result = resp.json()
print(result["prediction"])       # "Moderate"
print(result["confidence"])       # 0.82
print(result["referral_action"])  # "Refer to physician..."
```

### React Native (mobile app)

```typescript
const formData = new FormData();
formData.append('file', { uri: imageUri, type: 'image/jpeg', name: 'eye.jpg' } as any);
formData.append('age', String(ageMonths));
formData.append('gender', String(gender));
if (hbLevel) formData.append('hb_level', String(hbLevel));

const response = await axios.post('/predict/multimodal', formData, {
  headers: { 'Content-Type': 'multipart/form-data', 'X-API-Key': API_KEY },
});
```

### Example response

```json
{
  "prediction": "Moderate",
  "confidence": 0.8241,
  "confidence_score": 0.8241,
  "class_probabilities": {
    "Non-Anemic": 0.031,
    "Mild": 0.110,
    "Moderate": 0.824,
    "Severe": 0.035
  },
  "hb_estimate_gdl": 8.5,
  "risk_level": "high",
  "nutrition": "Increase iron-rich foods. Consider supplementation under medical supervision.",
  "recommended_foods": ["spinach", "lentils", "lean red meat", "fortified cereals"],
  "referral_action": "Refer to physician for evaluation within 1–2 weeks.",
  "recommendations": {
    "diet_plan": "Focus on iron-rich meals twice daily...",
    "foods_to_include": ["liver", "beans", "dark leafy greens"],
    "foods_to_avoid": ["tea with meals", "calcium-rich foods at iron meal times"],
    "urgency_level": "elevated"
  },
  "fusion_strategy": "tflite_fusion"
}
```

---

## Database Schema

Run `mobile/react_native_app/supabase_migration.sql` in the Supabase SQL Editor to create all tables, RLS policies, triggers, and storage buckets.

### Tables

| Table | Purpose | RLS scope |
|-------|---------|-----------|
| `public.profiles` | User roles and names (auto-created on sign-up) | Own row; admins see all via `is_admin()` |
| `public.screenings` | Prediction records | Own rows; admins see all |
| `public.children` | Child profiles linked to parent | Own rows |
| `public.sleep_logs` | Sleep tracking entries | Own rows |
| `public.feeding_logs` | Feeding tracking entries | Own rows |
| `public.analytics_events` | App usage events | Own rows |

### Key design decisions

- **Admin self-registration is blocked** — the `handle_new_user()` trigger rejects `role = 'admin'` from sign-up metadata. Admin accounts must be created manually in the Supabase dashboard.
- **`is_admin()` security definer** — avoids infinite RLS recursion when checking the `profiles` table from within a `profiles` policy.
- **Storage bucket** `conjunctiva-images` — private; objects scoped to `{user_id}/filename` path enforced by RLS.

---

## Environment Variables

### Backend (Railway / Docker)

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Yes | Shared secret for `X-API-Key` header authentication |
| `PORT` | Railway auto-set | Port uvicorn binds to |
| `PYTHONPATH` | Set in Dockerfile | `/app` — ensures module resolution |

### Mobile / Web (`.env` or EAS secrets)

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous public key |
| `API_BASE_URL` | FastAPI backend base URL |
| `API_KEY` | Key matching backend `API_KEY` |

---

## Running Tests

```bash
# From project root
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run specific suite
python -m pytest tests/test_inference.py -v
```

**Test suites (35 tests total):**

| File | Tests |
|------|-------|
| `test_inference.py` | TFLite prediction, fusion, weighted average, fallback logic |
| `test_api.py` | Live API endpoint contracts |
| `test_api_mocked.py` | Mocked image/multimodal prediction |
| `test_model.py` | RF model prediction and probabilities |
| `test_pipeline.py` | Full multimodal pipeline contract |
| `test_split_manifest.py` | Route existence, optional parameters |
| `test_dataset_integrity.py` | CSV structure, image/tabular pairing |
| `test_feature_schema.py` | Feature vector shapes and ordering |

---

## Generating Model Files

Run `Notebook/Bari.ipynb` sequentially in Google Colab or locally (Python 3.9–3.11 with TensorFlow).

| Notebook section | Output file | Used by |
|-----------------|-------------|---------|
| Random Forest (with HB) | `Notebook/models/tabular_with_hb.pkl` | `/predict/multimodal` (with HB) |
| Random Forest (no HB) | `Notebook/models/tabular_no_hb.pkl` | `/predict/multimodal` (no HB) |
| Visual TFLite export | `models/saved_models/visual_model.tflite` | `/predict/image` |
| Fusion TFLite export | `models/saved_models/multimodal_model.tflite` | `/predict/multimodal` (with HB) |
| Fusion TFLite export | `models/saved_models/multimodal_no_hb_model.tflite` | `/predict/multimodal` (no HB) |

Both `.pkl` files are joblib bundles: `{"model": RandomForestClassifier, "scaler": StandardScaler}`.

---

## Dataset Citation

Asare, Justice Williams; APPIAHENE, PETER; DONKOH, EMMANUEL (2023),
"CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana",
Mendeley Data, V1, doi: [10.17632/m53vz6b7fx.1](https://doi.org/10.17632/m53vz6b7fx.1)

---

## Clinical Disclaimer

**This system is for educational and research purposes only.**

- NOT a diagnostic tool — use only as a preliminary screening aid
- Always consult qualified healthcare professionals for diagnosis and treatment
- Developers assume no responsibility for medical decisions based on this system
- Must be used alongside, never instead of, proper clinical examination

---

## Contributors

- **Project:** Bari Anemia Screening System
- **Institution:** ALU — Capstone Project
- **Year:** 2026
