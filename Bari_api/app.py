# -------------------------
# app.api - Fusion Anemia Prediction with Nutrition Advice
# -------------------------

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import joblib
import uvicorn

# -------------------------
# 1. Initialize app
# -------------------------
app = FastAPI(
    title="Anemia Detection API",
    description="Predict anemia level from eyelid image + tabular data and provide WHO-based nutritional advice",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------
# 2. Load saved models
# -------------------------
# Tabular Random Forest
RF_model = joblib.load("random_forest_tabular_model.pkl")  

# Visual Fusion Model
fusion_model = tf.keras.models.load_model("fusion_model_tabular_visual.h5")

# Classes
classes = ['Anemic', 'Non-Anemic']

# -------------------------
# 3. Utility functions
# -------------------------
def preprocess_image(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    return np.expand_dims(img_array, axis=0)  # shape: (1,224,224,3)

def get_tabular_probs(age: float, gender: int, hb_level: float):
    X_tabular = np.array([[hb_level, age, gender]])  # match training order
    probs = RF_model.predict_proba(X_tabular)
    return probs  # shape: (1, num_classes)

def get_image_embeddings(img_array):
    # Extract embeddings from MobileNetV2 base (assumes fusion_model input shape)
    feature_extractor = tf.keras.models.Model(
        inputs=fusion_model.input, 
        outputs=fusion_model.layers[1].input  # assuming fusion concatenates after embeddings
    )
    embeddings = feature_extractor.predict(img_array)
    return embeddings

def get_fusion_input(img_embeddings, tabular_probs):
    return np.concatenate([img_embeddings, tabular_probs], axis=1)

def get_nutritional_advice(predicted_class, hb_level):
    if predicted_class == "Non-Anemic":
        return "Maintain a balanced diet with sufficient iron, folate, and vitamins."
    else:
        if hb_level >= 10:  # mild anemia example threshold
            return ("Mild anemia detected. Increase iron-rich foods (spinach, legumes, meat), "
                    "consume vitamin C to improve absorption, and follow regular check-ups.")
        else:
            return ("Moderate/severe anemia detected. Seek medical advice, "
                    "start iron supplements as advised, follow WHO anemia treatment guidelines.")

# -------------------------
# 4. Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict_anemia(
    image: UploadFile = File(...),
    age: float = Form(...),
    gender: int = Form(...),  # 0=Female, 1=Male
    hb_level: float = Form(...)
):
    try:
        # Preprocess image
        img_bytes = await image.read()
        img_array = preprocess_image(img_bytes)

        # Tabular probabilities
        tabular_probs = get_tabular_probs(age, gender, hb_level)

        # Image embeddings
        # Note: Adjust if you saved embeddings differently
        # For now, use MobileNetV2 directly
        image_embeddings = fusion_model.layers[1].input  # placeholder
        # fusion_input = get_fusion_input(image_embeddings, tabular_probs)
        # Simplified for API demonstration:
        fusion_input = np.concatenate([img_array.flatten()[None,:], tabular_probs], axis=1)

        # Predict
        fusion_preds = fusion_model.predict(fusion_input)
        pred_index = np.argmax(fusion_preds, axis=1)[0]
        confidence = float(fusion_preds[0, pred_index])
        anemia_class = classes[pred_index]

        # Nutrition advice
        advice = get_nutritional_advice(anemia_class, hb_level)

        return JSONResponse(content={
            "predicted_class": anemia_class,
            "confidence": confidence,
            "nutrition_advice": advice
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# -------------------------
# 5. Health check
# -------------------------
@app.get("/")
async def root():
    return {"message": "Anemia Detection API running successfully!"}

# -------------------------
# 6. Run server (for development)
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)