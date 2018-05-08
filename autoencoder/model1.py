from keras.models import Model
from keras.layers import Input
# for checkpoints
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,UpSampling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

def createModel(img_rows, img_cols, img_channels):
    input_img = Input(shape=(img_rows, img_cols, img_channels))  
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Define location to save the weights
    #checkpointer = ModelCheckpoint(filepath='/media/vahid/96CC807ACC805701/anthesis/fasterRCNNTassel2016/Matchfeatures/segmentation/Arabidopweights.hdf5', verbose=1, save_best_only=True)

    # Define optimizer, loss function and model performance metrics
    autoencoder = Model(input_img, output_img)
    return autoencoder
