from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import os
from utils import preprocess_image, read_image, predict_class

app = FastAPI()

# load trained model
model_path = os.path.join(os.path.dirname(__file__), "../Notebook/models/mobilenetv2_finetuned_visual_model.h5")
model = tf.keras.models.load_model(model_path)

classes = ["Anemic", "Non-Anemic"]

@app.get("/")
def home():
    return {"message": "Anemia Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image = await read_image(file)

    processed_image = preprocess_image(image)

    label, confidence = predict_class(model, processed_image, classes)

    return {
        "prediction": label,
        "confidence": confidence
    }