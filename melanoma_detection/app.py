import os
import sys
import cv2
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
from util import base64_to_pil


# Flask
app = Flask(__name__)

tf.config.experimental.set_visible_devices([], 'GPU')


# Model saved with Keras model.save()
MODEL_PATH = 'models/saved_model'

# Load trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          
print('Loaded model. Prediction page is openning..')
####################################################
class_names = ['melanoma', 'not melanoma'] 

def model_predict(img, model):
    img = img.resize((224, 224))
    img = np.reshape(img,[1,224,224,3])
    classes = np.argmax(model.predict(img), axis = -1)
    #print(classes)
    preds = [class_names[i] for i in classes]
    print("Sonuç: ", preds)
    return preds

###################################################

def model_predict1(img, model):
    img = img.resize((224, 224))

    # Preprocessing
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds 


@app.route('/', methods=['GET'])
def index():
    # Main
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get image
        img = base64_to_pil(request.json)

        # prediction
        preds = model_predict(img, model)
        print(preds)
        return jsonify(result=preds)

    return None


if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
