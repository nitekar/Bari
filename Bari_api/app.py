from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from utils import preprocess_image, make_gradcam_heatmap, get_nutrition_advice
import tensorflow as tf

app = FastAPI(title="Anemia Detection API")

# Load models
fusion_model = tf.keras.models.load_model("fusion_model.h5")
RF_model = joblib.load("RF_model.pkl")
classes = ["Normal", "Mild", "Moderate", "Severe"]  # Example classes

@app.post("/predict/")
async def predict_anemia(
    image: UploadFile,
    hb: float = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):
    # --- Preprocess image ---
    img_array = preprocess_image(await image.read())

    # --- Prepare tabular data ---
    gender_encoded = 1 if gender.lower() == "male" else 0
    X_tabular = np.array([[hb, age, gender_encoded]])
    tabular_probs = RF_model.predict_proba(X_tabular)

    # --- Extract image embeddings ---
    feature_extractor = tf.keras.models.Model(
        inputs=fusion_model.input,
        outputs=fusion_model.layers[-3].output
    )
    image_embeds = feature_extractor(img_array)

    # --- Fusion input ---
    fusion_input = np.concatenate([image_embeds.numpy(), tabular_probs], axis=1)
    pred_probs = fusion_model.predict(fusion_input)
    pred_class_idx = np.argmax(pred_probs)
    pred_class = classes[pred_class_idx]

    # --- Grad-CAM ---
    heatmap = make_gradcam_heatmap(img_array, fusion_model)
    heatmap_max = float(np.max(heatmap))

    # --- Nutritional advice ---
    advice = get_nutrition_advice(pred_class)

    return JSONResponse({
        "prediction": pred_class,
        "probabilities": pred_probs.tolist(),
        "gradcam_max_value": heatmap_max,
        "nutritional_advice": advice
    })