from __future__ import annotations

from pathlib import Path

import cv2
import imutils
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "cnn-parameters-improvement-23-0.91.h5",
    BASE_DIR / "models" / "cnn-parameters-improvement-23-0.91.h5",
    BASE_DIR / "models" / "cnn-parameters-improvement-04-0.63.h5",
]

IMG_WIDTH = 240
IMG_HEIGHT = 240
THRESHOLD = 0.5


def crop_brain_contour(image: np.ndarray) -> np.ndarray:
    """Crop the image around the largest contour to keep the brain area only."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)

    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bottom = tuple(c[c[:, :, 1].argmax()][0])

    cropped = image[ext_top[1] : ext_bottom[1], ext_left[0] : ext_right[0]]
    if cropped.size == 0:
        return image
    return cropped


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")

    image = crop_brain_contour(image)
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def build_model(input_shape: tuple[int, int, int]) -> Model:
    x_input = Input(input_shape)
    x = ZeroPadding2D((2, 2))(x_input)
    x = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(x)
    x = BatchNormalization(axis=3, name="bn0")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((4, 4), name="max_pool0")(x)
    x = MaxPooling2D((4, 4), name="max_pool1")(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid", name="fc")(x)
    return Model(inputs=x_input, outputs=x, name="BrainDetectionModel")


def load_best_model():
    model = build_model((IMG_WIDTH, IMG_HEIGHT, 3))
    errors = []
    for weights_path in MODEL_CANDIDATES:
        if weights_path.exists():
            try:
                model.load_weights(weights_path)
                return model
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{weights_path}: {exc}")

    candidates = "\n".join(str(path) for path in MODEL_CANDIDATES)
    if errors:
        error_details = "\n".join(errors)
        raise RuntimeError(
            f"Model files were found but could not be loaded. Tried:\n{candidates}\n\nErrors:\n{error_details}"
        )
    raise FileNotFoundError(f"No model file found. Tried:\n{candidates}")


app = Flask(__name__)
model = load_best_model()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    try:
        processed = preprocess_image(file.read())
        prob = float(model.predict(processed, verbose=0)[0][0])
        label = "Tumor Detected" if prob > THRESHOLD else "No Tumor"

        return jsonify(
            {
                "label": label,
                "probability": prob,
                "confidence_percent": round(prob * 100.0, 2),
                "threshold": THRESHOLD,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
