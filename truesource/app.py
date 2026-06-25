"""TrueSource AI - Flask backend serving a Keras deepfake image detector.

The model (deepfake_detector.keras) is loaded via model_setup.paths, which pulls
it from the HuggingFace Hub on first launch. Inference uses helpers.predict:
128x128 RGB input, single sigmoid output = P(AI-Generated).
"""
import io
import json
import os
import threading
import traceback

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

import model_setup
import helpers

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "sample_images")

with open(os.path.join(HERE, "class_names.json")) as f:
    CLASS_NAMES = tuple(json.load(f))  # ("Real", "AI-Generated")

app = Flask(__name__, static_folder="static", template_folder="templates")
# Cap request bodies to 16 MB to limit DoS exposure from large uploads/base64.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
# Guard against decompression-bomb images.
Image.MAX_IMAGE_PIXELS = 50_000_000

# --- Model loading state (loaded in background so the UI can show a loader) ---
_model = None
_model_error = None
_model_lock = threading.Lock()


def _load_model():
    global _model, _model_error
    try:
        import tensorflow as tf

        path = model_setup.paths["deepfake_detector.keras"]
        model = tf.keras.models.load_model(path)
        # Warm up so the first real prediction is fast.
        import numpy as np

        model.predict(np.zeros((1, helpers.IMG_SIZE, helpers.IMG_SIZE, 3), dtype="float32"), verbose=0)
        with _model_lock:
            _model = model
    except Exception as exc:  # noqa: BLE001
        with _model_lock:
            _model_error = str(exc)
        traceback.print_exc()


threading.Thread(target=_load_model, daemon=True).start()


def _list_samples():
    items = []
    if os.path.isdir(SAMPLE_DIR):
        for fname in sorted(os.listdir(SAMPLE_DIR)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            truth = "AI-Generated" if fname.lower().startswith("ai") else "Real"
            items.append({
                "name": fname,
                "truth": truth,
                "url": f"samples/{fname}",
            })
    return items


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/healthz")
def status():
    with _model_lock:
        ready = _model is not None
        error = _model_error
    return jsonify({"ready": ready, "error": error})


@app.route("/samples-list")
def samples():
    return jsonify(_list_samples())


@app.route("/samples/<path:filename>")
def sample_file(filename):
    return send_from_directory(SAMPLE_DIR, filename)


def _run_prediction(image: Image.Image):
    with _model_lock:
        model = _model
    if model is None:
        return None
    label, scores = helpers.predict(model, image, CLASS_NAMES)
    prob_ai = float(scores["AI-Generated"])
    prob_real = float(scores["Real"])
    return {
        "label": label,
        "prob_ai": prob_ai,
        "prob_real": prob_real,
        "confidence": max(prob_ai, prob_real),
    }


@app.route("/predict", methods=["POST"])
def predict():
    with _model_lock:
        if _model is None:
            if _model_error is not None:
                return jsonify({
                    "error": "model_failed",
                    "message": "The detection model failed to load. Please try again later.",
                }), 503
            return jsonify({"error": "model_loading", "message": "Model is still warming up."}), 503

    image = None
    try:
        if "file" in request.files:
            image = Image.open(request.files["file"].stream)
        elif request.is_json and request.json and request.json.get("sample"):
            sample = os.path.basename(request.json["sample"])
            path = os.path.join(SAMPLE_DIR, sample)
            if not os.path.exists(path):
                return jsonify({"error": "not_found", "message": "Sample not found."}), 404
            image = Image.open(path)
        elif request.is_json and request.json and request.json.get("image_data"):
            import base64

            data = request.json["image_data"]
            if "," in data:
                data = data.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(data)))
        else:
            return jsonify({"error": "no_input", "message": "No image provided."}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "bad_image", "message": f"Could not read image: {exc}"}), 400

    result = _run_prediction(image)
    if result is None:
        return jsonify({"error": "model_loading"}), 503
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
