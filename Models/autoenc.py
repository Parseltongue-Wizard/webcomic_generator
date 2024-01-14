import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

original_dim = 65536 # 256*256=65536 for the rezised comic images
intermediate_dim = 16 384
latent_dim = 2
