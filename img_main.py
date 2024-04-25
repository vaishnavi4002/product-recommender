import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential  # Ensure you import Sequential from keras.models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalMaxPooling2D

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained model
base_model.trainable = False

# Create a new Sequential model
model = Sequential([  # Use Sequential from keras.models
    base_model,
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

  
filenames = []

for file in os.listdir('downloaded_images'):
    filenames.append(os.path.join('downloaded_images',file))


feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))


pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))