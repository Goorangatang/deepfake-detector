import model_setup, json
import tensorflow as tf
m = tf.keras.models.load_model(model_setup.paths['deepfake_detector.keras'])
print('input shape:', m.input_shape, 'output shape:', m.output_shape)
import helpers
from PIL import Image
cn = tuple(json.load(open('class_names.json')))
for name in ['Real_0.png','Real_1.png','AI-Generated_0.png','AI-Generated_1.png']:
    img = Image.open(model_setup.paths[name])
    label, scores = helpers.predict(m, img, cn)
    print(name, '->', label, {k: round(v,3) for k,v in scores.items()})
