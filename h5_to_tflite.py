"""
convert h5 to tflite file
"""

import tensorflow as tf
from tensorflow import keras

model_name = input('Enter model name (in model dir, without extensions):')

model: keras.Sequential
with open('models/' + model_name + '.json', 'r') as json:
    model = keras.models.model_from_json(json.read())
model.load_weights('models/' + model_name + '.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_model = converter.convert()
open('models/' + model_name + '.tflite', 'wb').write(tf_model)
