from keras import objectives
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.core import Activation, Dropout
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.noise import GaussianNoise

def cae(settings):
    patch_size = settings["patch_size"]
    in_ch = settings["hyperparams"]["in_ch"]
    out_ch = settings["hyperparams"]["out_ch"]
    dropout_rate = settings["hyperparams"]["dropout_rate"]
    noise_rate = settings["hyperparams"]["stddev_noise"]

    input_img = Input(shape=(patch_size, patch_size, in_ch))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    pool1 = MaxPooling2D((2, 2), border_mode='same')(conv1)
    drop1 = Dropout(dropout_rate)(pool1)
    noise1 = GaussianNoise(noise_rate)(drop1)

    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(noise1)
    pool2 = MaxPooling2D((2, 2), border_mode='same')(conv2)
    drop2 = Dropout(dropout_rate)(pool2)
    noise2 = GaussianNoise(noise_rate)(drop2)

    conv3 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(noise2)
    encoded = MaxPooling2D((2, 2), border_mode='same')(conv3)

    conv4 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    pool4 = UpSampling2D((2, 2))(conv4)
    drop4 = Dropout(dropout_rate)(pool4)
    noise4 = GaussianNoise(noise_rate)(drop4)

    conv5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(noise4)
    pool5 = UpSampling2D((2, 2))(conv5)
    drop5 = Dropout(dropout_rate)(pool5)
    noise5 = GaussianNoise(noise_rate)(drop5)

    conv6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(noise5)
    pool6 = UpSampling2D((2, 2))(conv6)
    drop6 = Dropout(dropout_rate)(pool6)
    noise6 = GaussianNoise(noise_rate)(drop6)

    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(noise6)

    return Model(input_img, decoded)

def unet(settings):
    patch_size = settings["patch_size"]
    in_ch = settings["hyperparams"]["in_ch"]
    out_ch = settings["hyperparams"]["out_ch"]
    nf = settings["hyperparams"]["num_filt"]

    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }

    i = Input(shape=(self.in_ch, self.patch_size, self.patch_size))

    # in_ch x 512 x 512
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 4 x 4

    conv8 = Convolution(nf * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 2 x 2

    conv9 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8, (batch_size, nf*8, 2, 2), k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = merge([dconv1, conv8], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8, (batch_size, nf*8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = merge([dconv2, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8, (batch_size, nf*8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = merge([dconv3, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8, (batch_size, nf*8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = merge([dconv4, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8, (batch_size, nf*8, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = merge([dconv5, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4, (batch_size, nf*4, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = merge([dconv6, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2, (batch_size, nf*2, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = merge([dconv7, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf, (batch_size, nf, 256, 256))(x)
    dconv8 = BatchNorm()(dconv8)
    x = merge([dconv8, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(self.out_ch, (batch_size, self.out_ch, self.patch_size, self.patch_size))(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv9)

    return Model(i, out, name=name)

def pick_model(settings):
    if settings["model"] == "cae":
        return cae(settings)
    elif settings["model"] == "unet":
        return unet(settings)
