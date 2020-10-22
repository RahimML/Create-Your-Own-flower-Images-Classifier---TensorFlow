"""
$ pip install -q -U "tensorflow-gpu==2.0.0b1"
$ pip install -q -U tensorflow_hub
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
import argparse
from PIL import Image

def process_image(image):
    img = np.squeeze(image)
    img = tf.image.resize(img, (224, 224))
    img /= 255
    return img

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def predict(image_path, model, top_k ):
    im = Image.open(image_path)
    predict_image = np.asarray(im)
    img = process_image(predict_image)
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    values , indecies = tf.math.top_k(prediction, k= top_k)
    
    probs = values.numpy()[0]
    classes = [class_names[str(i+1)] for i in indecies.cpu().numpy()[0]]
    
    return probs, classes

Flowers_model = tf.keras.models.load_model('Flowers_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

parser = argparse.ArgumentParser()
parser.add_argument( '--top_k', default= 5, help = 'Enter top K ', type = int)
parser.add_argument( '--image_dir', default= './test_images/cautleya_spicata.jpg', help = 'Enter image path', type = str)
parser.add_argument( '--category_names', default= 'label_map.json', help = 'map label from json file of categories', type = str)


results = parser.parse_args()

top_k = results.top_k
image_path = results.image_dir
classes = results.category_names


def process_image(image):
    img = np.squeeze(image)
    img = tf.image.resize(img, (224, 224))
    img /= 255
    return img

with open(classes, 'r') as f:
    class_names = json.load(f)

def predict(image_path, model, top_k ):
    im = Image.open(image_path)
    predict_image = np.asarray(im)
    img = process_image(predict_image)
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    values , indecies = tf.math.top_k(prediction, k= top_k)
    
    probs = values.numpy()[0]
    classes = [class_names[str(i+1)] for i in indecies.cpu().numpy()[0]]
    
    return probs, classes

probs, classes = predict(image_path, Flowers_model, top_k)
print(classes)
print(probs)












