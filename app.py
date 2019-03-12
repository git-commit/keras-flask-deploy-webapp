from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json

# Keras (from tensorflow)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH_JSON = 'models/transfer-learning-resnet50.json'
MODEL_PATH_H5 = 'models/transfer-learning-resnet50.h5'

# Load your trained model
# json loading is done by hand here to prevent some errors with the way
# tensorflow-keras calls the python json parser 
with open(MODEL_PATH_JSON, 'r') as json_file:
    architecture = json.load(json_file)
    model = model_from_json(json.dumps(architecture))

model._make_predict_function()          # Necessary
model.load_weights(MODEL_PATH_H5)

class_associations = {0: 'defect', 1: 'good'}

print('Model loaded. Start serving on http://127.0.0.1:5000/...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1) # Simple argmax
        print(preds)
        print("Response from neural network (defect: {}; good: {})".format(preds[0][0], preds[0][1]))
        return str(class_associations[pred_class[0]])          
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
