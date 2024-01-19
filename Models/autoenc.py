import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
X_train, X_temp = train_test_split(normalized_images, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size = 0.5, random_state = 42)


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) *tf.exp(log_var / 2) + mean

latent_dim = 2

inputs = Input(shape=(256,256))
Z = Dense(150, activation='relu')(inputs)
Z = Dense(100, activation='relu')(Z)

codings_mean = tf.keras.layers.Dense(latent_dim)(Z) 
codings_log_var = tf.keras.layers.Dense(latent_dim)(Z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = tf.keras.layers.Input(shape=[latent_dim])
x = Dense(100, activation="relu")(decoder_inputs)
x = Dense(150, activation="relu")(x)
x = Dense(256*256)(x)
outputs = tf.keras.layers.Reshape([256,256])(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
    axis=-1
    )
variational_ae.add_loss(tf.reduce_mean(latent_loss) / float(original_dim))
variational_ae.compile(loss='mse', optimizer='nadam')
history = variational_ae.fit(X_train, X_train, epochs = 25, batch_size = 128, validation_data = (X_val, X_val))

def plot_visualisations(model, images=X_val, n_images=5):
    visualisations = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(visualisations[image_index], cmap="binary")
        plt.axis("off")
plot_visualisations(variational_ae)
plt.show()

