"""
convert h5 to tflite file
"""

import tensorflow as tf
from tensorflow import keras

model: keras.Sequential
with open('model_lichess.json', 'r') as json:
    model = keras.models.model_from_json(json.read())
model.load_weights('model_lichess.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_model = converter.convert()
open("model_lichess.tflite", "wb").write(tf_model)
