import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, Flatten, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


base_model=keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha = 0.75,depth_multiplier = 1, dropout = 0.001, pooling='avg',include_top = True, weights = "imagenet", classes = 1000)
params = np.array(base_model.get_weights())
np.savez('mbnetv1_10.75.npz', params=params)
