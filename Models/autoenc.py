import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

original_dim = 65536 # 256*256=65536 for the rezised comic images
intermediate_dim = 16 384
latent_dim = 2

def load_and_normalize_images(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            # Convert image to NumPy array and normalize pixel values
            img_array = np.array(img) / 255.0
            images.append(img_array)

    return images

folder_path = "Resized_data"
normalized_images = load_and_normalize_images(folder_path)
X_train, X_test = train_test_split(normalized_images, test_size=0.1, random_state=42)
