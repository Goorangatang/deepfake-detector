"""Saliency / Grad-CAM heatmap generation for the deepfake detector.

Produces a heatmap overlay highlighting the image regions that most influenced
the model's "AI-Generated" likelihood. Uses Grad-CAM on the last convolutional
feature map when available, and falls back to input-gradient saliency for
architectures without an obvious conv layer. Returns a base64 PNG data URL so it
can be dropped straight into an <img> on the frontend.
"""
import base64
import io

import numpy as np
from PIL import Image

import helpers


def _find_last_conv_layer(model):
    """Return the name of the last layer with a 4D (H, W, C) output."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:  # noqa: BLE001
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    return None


def _ai_score(preds):
    import tensorflow as tf

    preds = tf.reshape(preds, [tf.shape(preds)[0], -1])
    return preds[:, 0]


_grad_models = {}


def _get_grad_model(model, layer_name):
    import tensorflow as tf

    key = (id(model), layer_name)
    grad_model = _grad_models.get(key)
    if grad_model is None:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(layer_name).output, model.output]
        )
        _grad_models[key] = grad_model
    return grad_model


def _gradcam(model, arr, layer_name):
    import tensorflow as tf

    grad_model = _get_grad_model(model, layer_name)
    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = _ai_score(preds)
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heat = tf.reduce_sum(conv_out * pooled, axis=-1)
    heat = tf.nn.relu(heat)
    return heat.numpy()


def _grad_saliency(model, arr):
    import tensorflow as tf

    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        loss = _ai_score(preds)
    grads = tape.gradient(loss, x)[0].numpy()
    return np.max(np.abs(grads), axis=-1)


def _jet(x):
    """Classic jet colormap. Input HxW in [0,1] -> HxWx3 in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _normalize(heat):
    heat = heat.astype("float32")
    heat -= heat.min()
    peak = heat.max()
    if peak > 1e-8:
        heat /= peak
    else:
        heat = np.zeros_like(heat)
    return heat


def _to_overlay(orig_img, heat, size=320):
    base = orig_img.convert("RGB").resize((size, size))
    base_arr = np.asarray(base).astype("float32")

    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize(
        (size, size), Image.BILINEAR
    )
    heat_resized = np.asarray(heat_img).astype("float32") / 255.0

    color = _jet(heat_resized) * 255.0
    alpha = (heat_resized ** 0.8)[..., None] * 0.6
    out = base_arr * (1 - alpha) + color * alpha
    out = np.clip(out, 0, 255).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saliency_overlay(model, image):
    """Return a base64 PNG data URL heatmap overlay, or None on failure."""
    arr = helpers.preprocess(image)

    heat = None
    layer_name = _find_last_conv_layer(model)
    if layer_name is not None:
        try:
            heat = _gradcam(model, arr, layer_name)
        except Exception:  # noqa: BLE001
            heat = None
    if heat is None:
        heat = _grad_saliency(model, arr)

    heat = _normalize(heat)
    return _to_overlay(image, heat)
