import numpy as np
import tensorflow as tf
import cv2

def preprocess_image(img_bytes, target_size=(224,224)):
    # Read bytes and convert to image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1', pred_index=None):
    # Access the MobileNetV2 inside the model
    base_model_layer = model.get_layer('mobilenetv2_1.00_224')
    conv_layer = base_model_layer.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_nutrition_advice(anemia_class):
    advice_dict = {
        "Normal": "Maintain a balanced diet with iron-rich foods.",
        "Mild": "Include iron-rich foods like spinach, lentils, lean meat. Avoid tea/coffee with meals.",
        "Moderate": "Increase iron + vitamin C intake. Monitor symptoms and consult a doctor if persistent.",
        "Severe": "Immediate consultation with a doctor is advised. Follow medical advice and iron supplements."
    }
    return advice_dict.get(anemia_class, "Follow a balanced diet.")