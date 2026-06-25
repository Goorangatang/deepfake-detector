# TruthLens deepfake detector - preprocessing + inference helper.
# The model expects 128x128 RGB images normalized to 0-1 (divide by 255).
# It outputs ONE sigmoid value = P(AI-Generated). class_names = ["Real", "AI-Generated"].
# Label is "AI-Generated" if prob >= 0.5 else "Real". Do NOT argmax a single-value output.
import numpy as np
from PIL import Image

IMG_SIZE = 128

def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def predict(model, image, class_names=("Real", "AI-Generated")):
    prob_ai = float(model.predict(preprocess(image), verbose=0).flatten()[0])
    scores = {class_names[0]: 1.0 - prob_ai, class_names[1]: prob_ai}
    label = class_names[1] if prob_ai >= 0.5 else class_names[0]
    return label, scores
