from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

app = Flask(__name__)
model = load_model("best_model.keras")
class_names = ["Glass", "Plastic"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "缺少圖片"})

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((299, 299))
    img_array = np.expand_dims(np.array(image), axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array, verbose=0)[0]
    class_index = np.argmax(pred)
    confidence = float(pred[class_index])
    label = class_names[class_index]

    return jsonify({
        "label": label,
        "confidence": round(confidence, 3)
    })