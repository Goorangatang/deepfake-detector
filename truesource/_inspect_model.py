import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import model_setup, tensorflow as tf
m = tf.keras.models.load_model(model_setup.paths["deepfake_detector.keras"])
print("INPUT", m.input_shape, "OUTPUT", m.output_shape)
convs = []
for l in m.layers:
    try:
        sh = l.output.shape
    except Exception:
        sh = None
    print(l.name, type(l).__name__, sh)
    if "Conv" in type(l).__name__:
        convs.append(l.name)
print("CONV LAYERS:", convs)
