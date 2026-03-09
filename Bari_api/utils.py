import numpy as np
from PIL import Image
import io

IMG_SIZE = 224

async def read_image(file):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    return image


def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


def predict_class(model, image, classes):

    prediction = model.predict(image)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    label = classes[predicted_index]

    return label, round(confidence, 3)