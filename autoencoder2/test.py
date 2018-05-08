from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from patcher import Patcher
import h5py
from PIL import ImageFilter, Image
from models import *
import scipy.stats
import os

import utils
settings = utils.settings

model = pick_model(settings)
patch_size = settings['patch_size']

print(model.summary())
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.load_weights(settings['weights_file'])

def predictor(_patches):
    return model.predict(_patches)

imgs = os.listdir(settings['test']['img_path'])

preds = []

print('Predicting labels...')
for img in imgs:
    print(img)
    img_file = settings['test']['img_path'] + '/' + img
    patcher_test = Patcher.for_test(img_file, _dim=(patch_size, patch_size))

    pred_label = patcher_test.predict(predictor)
    # pred_label = pred_label / 9.0
    pred_label = np.clip(pred_label, 0.0, 1.0)

    vfunc = np.vectorize(lambda v: 1 if v > 0.1 else 0)
    pred_label = vfunc(pred_label)

    preds.append(pred_label)
    data = np.asarray(preds[0])
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('test.png')
    print (data.shape)

