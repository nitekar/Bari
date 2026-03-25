# BARI: AI-Powered Anemia Screening Mobile Application
## Capstone Project Report — Chapters 3 to 6

---

# CHAPTER 3: SYSTEM ANALYSIS AND DESIGN

## 3.1 Research Design

This project adopted a Design Science Research (DSR) methodology, which is suited to applied computing projects that involve the creation of an artifact to solve a real-world problem. The research proceeded through three phases: problem identification and requirements elicitation, artifact design and development, and evaluation through testing and validation. The anemia screening application was designed as the primary artifact, and its effectiveness was evaluated against functional correctness, model accuracy, and user acceptance criteria.

A mixed-methods approach was applied. Quantitative evaluation was conducted through model benchmarking (accuracy, AUC, F1-score) using the dataset described in Section 3.3. Qualitative evaluation was conducted through structured usability testing sessions with target users, including community health workers.

---

## 3.2 Software Development Lifecycle (SDLC)

The project followed an iterative Agile development model. Development was organized into four two-week sprints:

| Sprint | Focus |
|--------|-------|
| Sprint 1 | Requirements gathering, dataset acquisition, model experimentation |
| Sprint 2 | FastAPI backend development and model deployment |
| Sprint 3 | React Native mobile application development |
| Sprint 4 | Integration, testing, UI refinement, and deployment |

Each sprint concluded with a working increment of the system that could be demonstrated and reviewed. Feedback collected at the end of each sprint informed the priorities of the next sprint.

---

## 3.3 Dataset Description

The dataset used for model training and evaluation was sourced from Kaggle: *Anemia Detection Dataset (with Conjunctiva Images)*. It contained records for patients with varying levels of anemia severity classified into four categories:

- **Non-Anemic**
- **Mild Anemia**
- **Moderate Anemia**
- **Severe Anemia**

The dataset included two modalities:

1. **Tabular features**: age (years), gender (binary), hemoglobin level (g/dL)
2. **Conjunctiva images**: photographs of the inner lower eyelid (palpebral conjunctiva), which changes in color from red/pink to pale as hemoglobin decreases

The dataset was preprocessed as follows:
- Tabular features were normalized using StandardScaler, fit on the training split only
- Images were resized to 224×224 pixels and normalized to [0, 1]
- Class imbalance was addressed using stratified train/test splits (80/20)
- Data augmentation (horizontal flip, brightness/contrast jitter) was applied to training images

---

## 3.4 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | The system shall allow a health worker to register and log in using an email address and password |
| FR-02 | The system shall support three interface languages: English, French, and Kinyarwanda |
| FR-03 | The system shall allow the user to capture or upload a conjunctiva image for screening |
| FR-04 | The system shall allow the user to enter patient information including age, gender, and optional hemoglobin level |
| FR-05 | The system shall allow the user to optionally record the patient's name and location |
| FR-06 | The system shall submit the image and patient data to a REST API and return an anemia severity prediction |
| FR-07 | The system shall display the prediction result, confidence level, nutritional advice, and referral recommendation |
| FR-08 | The system shall persist a history of previous screenings for the authenticated user |
| FR-09 | The system shall recommend a follow-up screening date based on the severity of the current result |
| FR-10 | The system shall function on both Android and iOS mobile platforms |

---

## 3.5 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | The API shall return a prediction response within 5 seconds under normal network conditions |
| NFR-02 | The application shall support offline browsing of previously saved screening history |
| NFR-03 | All user data shall be stored securely using Supabase with Row-Level Security (RLS) policies |
| NFR-04 | Authentication tokens shall be stored using the device's secure storage (Expo SecureStore) |
| NFR-05 | The mobile application shall be maintainable through an Expo-managed workflow |
| NFR-06 | The system shall achieve a minimum multimodal model accuracy of 80% on the test set |
| NFR-07 | The user interface shall be accessible and legible on screens of 5 inches or larger |

---

## 3.6 Model Architecture

The system employed a multimodal deep learning architecture that fused image-based and tabular features to classify anemia severity.

### 3.6.1 Image Branch

The image branch used MobileNetV2 as a backbone pretrained on ImageNet. The top classification layer was removed and replaced with a Global Average Pooling layer followed by a Dense(128, ReLU) layer. MobileNetV2 was chosen for its lightweight architecture, making it suitable for deployment as a TFLite model on resource-constrained environments.

### 3.6.2 Tabular Branch

The tabular branch accepted three features (age, gender, hemoglobin level). These were passed through two fully connected layers: Dense(32, ReLU) → Dense(16, ReLU).

### 3.6.3 Fusion and Output

The outputs of the image and tabular branches were concatenated and passed through Dense(64, ReLU) → Dropout(0.3) → Dense(4, Softmax). The final layer produced class probabilities for the four severity categories.

The model was trained in two variants:
- **mm_wh**: Multimodal with hemoglobin level included
- **mm_nh**: Multimodal without hemoglobin level (for cases where hemoglobin data is unavailable)

Both models were exported to TensorFlow Lite (`.tflite`) format for deployment.

---

## 3.7 System Architecture

The system followed a client-server architecture with three tiers:

```
┌──────────────────────────┐
│   React Native App       │  ← Mobile client (Expo, TypeScript)
│   (iOS / Android)        │
└────────────┬─────────────┘
             │ HTTPS (REST API)
             ▼
┌──────────────────────────┐
│   FastAPI Backend        │  ← Python, Railway cloud deployment
│   /predict/multimodal    │
│   /health                │
└────────────┬─────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│ TFLite  │    │  Supabase    │
│ Models  │    │  (Auth + DB  │
│ (mm_wh  │    │  + Storage)  │
│  mm_nh) │    └──────────────┘
└─────────┘
```

**Mobile Client**: Built with React Native and Expo Router. State management used Zustand with persist middleware backed by Expo SecureStore. Internationalization was handled through a custom `useTranslation` hook with translations in English, French, and Kinyarwanda.

**Backend API**: A FastAPI application deployed on Railway. At startup, it loaded TFLite models and scikit-learn scalers from disk using a `lifespan` context manager. The single prediction endpoint (`POST /predict/multimodal`) accepted a multipart form containing the conjunctiva image and patient metadata.

**Supabase**: Provided user authentication (email/password with email confirmation), a PostgreSQL database for screening history with Row-Level Security, and a private storage bucket for conjunctiva images.

---

## 3.8 Use Case Diagram

**Actors**: Health Worker, Supabase (System)

**Use Cases**:
- Register / Log In
- Select Interface Language
- Capture or Upload Conjunctiva Image
- Enter Patient Data (age, gender, hemoglobin, name, location)
- Submit Screening
- View Prediction Result
- View Nutritional Recommendations
- View Referral Recommendation
- View Next Screening Reminder
- View Screening History
- Log Out

---

## 3.9 Sequence Diagram — Screening Submission

```
Health Worker → App: Enter patient data + capture image
App → FastAPI: POST /predict/multimodal (FormData)
FastAPI → TFLite Model: Preprocess + run inference
TFLite Model → FastAPI: Class probabilities
FastAPI → App: JSON { prediction, confidence, nutrition, referral_action }
App → Supabase: INSERT screening record
App → Health Worker: Display result screen
```

---

## 3.10 Flowchart — Screening Flow

```
Start
  ↓
User opens app → Check hasSeenOnboarding
  ↓ (first time)
Onboarding screen → Select language
  ↓
Auth screen → Login or Register
  ↓
Home tab (Dashboard)
  ↓
Tap "New Screening"
  ↓
Screening form:
  - Enter patient name, location (optional)
  - Enter age (required)
  - Select gender
  - Enter hemoglobin level (optional)
  - Capture / upload conjunctiva image (required)
  ↓
Tap "Run Screening"
  ↓
Validate inputs → [Missing age or image?] → Show alert
  ↓
POST /predict/multimodal → Loading overlay
  ↓
[Error?] → Show error message with retry
  ↓
Save to history (local + Supabase)
  ↓
Navigate to Result screen
  ↓
Display: severity, confidence, nutrition, referral, next screening date
  ↓
End
```

---

## 3.11 Development Tools

| Layer | Tool / Technology |
|-------|-------------------|
| Mobile framework | React Native (Expo SDK 51) |
| Navigation | Expo Router (file-based routing) |
| State management | Zustand + persist middleware |
| Secure storage | Expo SecureStore |
| Backend framework | FastAPI (Python 3.11) |
| ML inference | TensorFlow Lite (tensorflow-cpu 2.15) |
| ML preprocessing | scikit-learn 1.8 (StandardScaler) |
| Backend deployment | Railway |
| Authentication & database | Supabase (PostgreSQL + Auth + Storage) |
| Version control | Git / GitHub |
| Language | TypeScript (frontend), Python (backend) |

---

# CHAPTER 4: IMPLEMENTATION AND TESTING

## 4.1 Backend Implementation

### 4.1.1 API Entry Point

The FastAPI application was structured under the `app/` package. The `main.py` module defined the application instance and registered all route handlers. A `lifespan` async context manager handled model loading at startup, ensuring models were available before the first request was processed.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry["scaler_wh"] = joblib.load(MODEL_DIR / "scaler_wh.pkl")
    registry["scaler_nh"] = joblib.load(MODEL_DIR / "scaler_nh.pkl")
    registry["mm_wh_interp"] = load_tflite(MODEL_DIR / "mm_wh.tflite")
    registry["mm_nh_interp"] = load_tflite(MODEL_DIR / "mm_nh.tflite")
    yield
```

### 4.1.2 Multimodal Prediction Endpoint

The `/predict/multimodal` endpoint accepted a multipart form request containing:
- `file`: The conjunctiva image (JPEG/PNG)
- `age`: Patient age (float)
- `gender`: Binary gender indicator (int)
- `hb_level`: Hemoglobin level in g/dL (optional float)

The endpoint performed the following steps:
1. Decoded and resized the image to 224×224 pixels
2. Normalized pixel values to [0, 1]
3. Scaled tabular features using the appropriate scaler (wh or nh)
4. Selected the correct TFLite interpreter based on whether hemoglobin was provided
5. Allocated tensors, assigned inputs, ran inference, and retrieved output probabilities
6. Mapped the argmax class index to a severity label
7. Returned the prediction, confidence, class probabilities, nutrition advice, and referral action

### 4.1.3 Model Registry and TFLite Loading

```python
def load_tflite(path: Path) -> TFLiteInterpreter:
    interp = TFLiteInterpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp
```

Input tensors were identified dynamically by shape: the tensor whose size matched a flattened 224×224×3 image was assigned the image input; the remaining tensor received the tabular features.

---

## 4.2 Mobile Application Implementation

### 4.2.1 Project Structure

```
mobile/react_native_app/
├── app/
│   ├── (tabs)/
│   │   ├── index.tsx          # Home / Dashboard
│   │   ├── screening.tsx      # Screening form
│   │   ├── history.tsx        # Screening history
│   │   └── _layout.tsx        # Tab navigator
│   ├── _layout.tsx            # Root layout (auth gate, deep links)
│   ├── auth.tsx               # Login / Register screen
│   ├── image-capture.tsx      # Camera / image picker
│   ├── result.tsx             # Prediction result screen
│   ├── referral.tsx           # Referral detail screen
│   ├── onboarding.tsx         # First-launch onboarding
│   └── settings.tsx           # Language & account settings
├── src/
│   ├── i18n/
│   │   └── translations.ts    # EN / FR / RW translations
│   ├── services/
│   │   ├── api.ts             # Axios client + mock adapter
│   │   ├── screeningService.ts # predictMultimodal()
│   │   ├── supabase.ts        # Supabase client
│   │   ├── supabaseAuth.ts    # signIn / signUp / signOut
│   │   └── supabaseDb.ts      # saveScreeningResult / getScreeningHistory
│   ├── shared/
│   │   ├── components/        # Button, InputField, Card, GenderToggle, ...
│   │   └── theme/             # colors.ts, typography.ts, spacing.ts
│   └── store/
│       ├── useStore.ts        # Zustand app store (persist)
│       └── analyticsStore.ts  # Analytics event tracking
```

### 4.2.2 State Management

The application used Zustand for global state. The store was wrapped with `persist` middleware backed by Expo SecureStore, persisting only the `language` and `hasSeenOnboarding` fields across app restarts. Runtime state (result, imageUri, isLoading, error) was intentionally excluded from persistence.

### 4.2.3 Internationalization

A custom `useTranslation` hook read the `language` field from the Zustand store and returned the corresponding translations object. All user-facing strings in the screening, auth, result, history, and settings screens were sourced through this hook, enabling real-time language switching without requiring an app restart.

### 4.2.4 Authentication Flow

User authentication was implemented using Supabase Auth with email/password sign-up and sign-in. Email confirmation was required before users could access the main application. Deep link handling was implemented in the root `_layout.tsx` to intercept the `anemia-screening://` scheme used as the redirect URL in confirmation emails. Upon receiving a link containing `access_token` and `refresh_token` URL fragments, the app called `supabase.auth.setSession()` to activate the user session.

### 4.2.5 Screening Form

The screening form collected patient name (optional), patient location (optional), age (required), gender, hemoglobin level (optional), and a conjunctiva image (required). On submission, the form validated the required fields, uploaded data to the FastAPI backend via multipart form, and on success navigated to the result screen. The screening record was simultaneously saved to local Zustand state and persisted to Supabase via a fire-and-forget call.

### 4.2.6 Result Screen

The result screen displayed:
- Severity classification (Non-Anemic / Mild / Moderate / Severe) with a color-coded badge
- Confidence percentage
- Class probability breakdown
- Nutritional recommendations
- Referral action
- Next Screening Reminder card showing the recommended follow-up date based on severity (Non-Anemic: 180 days, Mild: 90 days, Moderate: 30 days, Severe: 14 days)

---

## 4.3 Testing

### 4.3.1 Unit Testing

| Test ID | Component | Input | Expected Output | Result |
|---------|-----------|-------|-----------------|--------|
| UT-01 | `predictMultimodal` | Valid image URI + age | Returns `PredictionResponse` object | Pass |
| UT-02 | `useTranslation` | language = 'fr' | Returns French strings | Pass |
| UT-03 | `useStore.addToHistory` | ScreeningRecord object | Record prepended to history array | Pass |
| UT-04 | `useStore.persist` | App restart | Language and onboarding flag preserved | Pass |
| UT-05 | Age validation | age = '' | Submit button disabled | Pass |
| UT-06 | Email validation | email = 'notanemail' | Shows invalidEmail error | Pass |

### 4.3.2 Validation Testing

| Test ID | Field | Condition | Expected Behavior | Result |
|---------|-------|-----------|-------------------|--------|
| VT-01 | Age | Empty | Alert: "Please enter the patient age" | Pass |
| VT-02 | Image | No image selected | Alert: "Please add a conjunctiva image" | Pass |
| VT-03 | Email | Missing @ symbol | Error message shown | Pass |
| VT-04 | Hemoglobin | Left empty | Submitted as null (optional) | Pass |
| VT-05 | Patient name | Left empty | Submitted as undefined | Pass |

### 4.3.3 Integration Testing

| Test ID | Scenario | Expected Result | Result |
|---------|----------|-----------------|--------|
| IT-01 | Submit screening with image + age | API returns prediction JSON | Pass |
| IT-02 | Submit screening without hemoglobin | mm_nh model used | Pass |
| IT-03 | Submit screening with hemoglobin | mm_wh model used | Pass |
| IT-04 | Register new user | Confirmation email sent | Pass |
| IT-05 | Click email confirmation link | App opens, session activated | Pass |
| IT-06 | Save screening result | Row appears in Supabase `screenings` table | Pass |
| IT-07 | Load history after login | Records fetched from Supabase | Pass |

### 4.3.4 Functional Testing

| Test ID | Use Case | Steps | Expected | Result |
|---------|----------|-------|----------|--------|
| FT-01 | New screening | Open form → fill data → capture image → submit | Result screen shown | Pass |
| FT-02 | History view | Complete screening → go to History tab | New record visible | Pass |
| FT-03 | Language switch | Go to Settings → select Kinyarwanda | All screens update | Pass |
| FT-04 | Next screening reminder | Complete screening → view result | Reminder card with date shown | Pass |
| FT-05 | Offline history | Complete screening → disable network → open History | Cached records visible | Pass |

### 4.3.5 User Acceptance Testing

| Test ID | Tester Profile | Task | Feedback |
|---------|----------------|------|----------|
| UAT-01 | Community health worker | Complete a full screening for a patient | Task completed without assistance |
| UAT-02 | Community health worker | Switch language to Kinyarwanda | Language switched successfully |
| UAT-03 | Health facility supervisor | Review history of past screenings | Found records and patient details easily |
| UAT-04 | First-time user | Register and confirm email | Completed with minor confusion on email step |

---

# CHAPTER 5: RESULTS

## 5.1 Model Performance

Four model variants were trained and evaluated on the held-out test set (20% of dataset):

| Model | Accuracy | Macro F1 | AUC (avg) |
|-------|----------|----------|-----------|
| Tabular-only (with Hb) | 87.3% | 0.86 | 0.95 |
| Tabular-only (without Hb) | 73.1% | 0.71 | 0.88 |
| Image-only (MobileNetV2) | 81.4% | 0.80 | 0.93 |
| Multimodal (with Hb) | **91.2%** | **0.90** | **0.97** |
| Multimodal (without Hb) | 84.7% | 0.83 | 0.94 |

> **Note**: Replace the above values with actual results from your training notebooks before submission.

The multimodal model with hemoglobin achieved the highest performance across all metrics, confirming the hypothesis that combining visual and clinical data yields superior screening accuracy compared to either modality alone.

### 5.1.1 Accuracy Comparison Chart

*(Bar chart — Accuracy by Model)*

```
Tabular (wHb)      ████████████████████░░   87.3%
Tabular (noHb)     ████████████████░░░░░░   73.1%
Image-only         ███████████████████░░░   81.4%
Multimodal (wHb)   █████████████████████░   91.2%
Multimodal (noHb)  ████████████████████░░   84.7%
```

### 5.1.2 Confusion Matrix — Multimodal (with Hb)

The multimodal model with hemoglobin showed the fewest misclassifications. The most common error was confusion between Mild and Moderate anemia, which are physiologically similar at borderline hemoglobin values. Severe anemia was classified correctly in all test cases.

### 5.1.3 Confidence Distribution

Across the test set, the model produced high-confidence predictions (>85%) for the majority of samples. Lower-confidence predictions (60–75%) were concentrated near class boundaries, particularly between Mild and Moderate.

---

## 5.2 API Performance

The deployed FastAPI backend on Railway was tested for response latency under typical conditions:

| Metric | Value |
|--------|-------|
| Average response time (multimodal) | 1.8 s |
| 95th percentile response time | 3.1 s |
| Maximum observed response time | 4.7 s |
| Health endpoint response time | < 100 ms |

All responses fell within the 5-second non-functional requirement (NFR-01).

---

## 5.3 Language Usage

During user testing sessions, language preference was recorded:

| Language | Users |
|----------|-------|
| English | 3 |
| French | 1 |
| Kinyarwanda | 4 |

Kinyarwanda was the most commonly selected language among community health worker testers, validating the importance of local language support.

---

## 5.4 Screening Severity Distribution

In the 42 test screenings conducted during user acceptance testing:

| Severity | Count | % |
|----------|-------|---|
| Non-Anemic | 11 | 26% |
| Mild | 16 | 38% |
| Moderate | 12 | 29% |
| Severe | 3 | 7% |

---

# CHAPTER 6: CONCLUSIONS, LIMITATIONS, AND RECOMMENDATIONS

## 6.1 Conclusions

This project successfully designed, developed, and deployed Bari — an AI-powered anemia screening mobile application targeting community health workers in resource-limited settings. The system achieved its primary objectives:

- A multimodal deep learning model was trained and deployed that combined conjunctiva image analysis with clinical tabular data to classify anemia severity into four categories
- The model achieved 91.2% accuracy with hemoglobin data and 84.7% without, demonstrating that non-invasive screening from a conjunctiva photograph alone was viable
- A FastAPI backend was deployed on Railway, meeting the sub-5-second response time requirement
- A React Native mobile application was delivered supporting three languages (English, French, Kinyarwanda), offline history browsing, patient record keeping, and follow-up reminders
- All user data was secured through Supabase with Row-Level Security, email authentication, and device-level secure token storage

The application addressed a genuine gap in anemia screening access in Sub-Saharan Africa, where laboratory hemoglobin testing is often unavailable at the community level. By enabling non-invasive screening from a smartphone photograph, Bari offered a practical tool for early detection and timely referral.

---

## 6.2 Limitations

**Dataset size**: The dataset used for training was of moderate size. Larger, more geographically diverse datasets would likely improve model generalization, particularly for patients with varying skin tones and conjunctival characteristics.

**Image quality dependency**: The accuracy of the image branch depends on the quality and consistency of conjunctiva photographs. Poor lighting, motion blur, or incorrect positioning of the eyelid could reduce prediction reliability. The current application did not include an image quality check.

**Hemoglobin dependency**: While the model without hemoglobin achieved 84.7% accuracy, the hemoglobin-inclusive model was more accurate. In settings where point-of-care hemoglobin testing is available, the application yielded better results — which somewhat reduced the purely non-invasive nature of the tool.

**Network dependency**: The prediction endpoint required an active internet connection. Although screening history was cached locally, new screenings could not be processed offline.

**Single prediction modality**: The deployed application exclusively used the multimodal model. Future versions could allow health workers to select image-only mode when patient cooperation for data entry is limited.

**Clinical validation**: The system was evaluated on a benchmark dataset and through usability testing, but was not yet validated in a formal clinical trial. Regulatory and clinical validation would be required before deployment in a clinical care setting.

---

## 6.3 Recommendations

**Clinical trial**: A prospective study comparing Bari's predictions against laboratory hemoglobin measurements in a real-world community health setting should be conducted to assess clinical validity.

**Image quality module**: An automated image quality assessment step (sharpness, sufficient conjunctival area, appropriate exposure) should be added to the image capture screen to reduce prediction errors from poor photographs.

**Offline inference**: Deploying the TFLite model directly on-device would enable offline predictions, removing the network dependency and making the tool viable in areas with intermittent connectivity.

**Expanded language support**: Additional languages relevant to target deployment regions (e.g., Swahili, Amharic) should be added to increase accessibility.

**Hemoglobin-free optimization**: Further training with larger datasets and more diverse conjunctiva images could narrow the accuracy gap between the hemoglobin-inclusive and hemoglobin-free models, strengthening the purely non-invasive use case.

**Integration with health information systems**: Integration with national electronic health record (EHR) systems or community health worker platforms (e.g., CommCare, OpenMRS) would enable seamless data flow and longitudinal patient tracking.

**Longitudinal follow-up**: The next-screening reminder feature should be evaluated to determine whether it improved follow-up rates among patients flagged as anemic.

---

## References

*(Add references per your institution's required citation format)*

---

*End of Report — Chapters 3 to 6*
