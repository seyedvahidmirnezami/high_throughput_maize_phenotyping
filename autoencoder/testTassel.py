# coding: utf-8
import utils

settings = utils.settings
import sys
name_model = sys.argv[1].split('.')[0];
print(name_model)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
from scipy.misc import imread
from scipy.misc import imresize
from keras import metrics
#from __future__ import print_function
from keras.models import Model
from keras.layers import Input
# for checkpoints
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,UpSampling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
#from createModel import createModel
def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)
moduleNames = [name_model] 
createModel = import_from(name_model, 'createModel')

warnings.filterwarnings('ignore')



# Define the data source
inputimagedata = settings['data_path_test']
#inputimagedata='/media/vahid/96CC807ACC805701/kevin2018/segmentationandRCNN/ParentChildRootImages/segnet/input'
#inputimagedata=inputimagedata.replace('\\', '/')
maskedimagedata = settings['data_path_test']
#maskedimagedata='/media/vahid/96CC807ACC805701/kevin2018/segmentationandRCNN/ParentChildRootImages/segnet/label'
#maskedimagedata=maskedimagedata.replace('\\','/')
import os
cwd = os.getcwd()

import glob
num_images = len(glob.glob(inputimagedata + '/*.' + settings['train']['img_ext']));
#name_model = settings['name_model'];

'''
filelist=os.listdir(inputimagedata)
image_name = filelist[0].split('.')[0]
imagepath= inputimagedata + '/' + image_name + '.' + settings['train']['img_ext']
print(imagepath)
img=imread(imagepath)
img1=imresize(img,(256,2556,3));
img1 = img1.astype('float32') / 255.0
I2=img1.flatten()
print('***')
#plt.imshow(imresize(I2,256,256,3))
plt.show()
'''
try:
    os.stat(name_model)
except:
    os.mkdir(name_model)       

patch_size = settings['patch_size'];
img_size = settings['image_size'];

#s=1000;
# Define the preprocessing algorithm
def downsample_flatten1(imagefoldername,noimage,nchannels,patch_size,img_size):
    print(patch_size)
    if patch_size != 0:
        test_input=np.empty([noimage,patch_size*patch_size*nchannels])
        test_ouput=np.empty([noimage,patch_size*patch_size*1])
    else:
        test_input=np.empty([noimage,img_size*img_size*nchannels])
        test_ouput=np.empty([noimage,img_size*img_size*1])


    
    train_input_name = [];
    filelist=os.listdir(imagefoldername)
    count=0
    for file in filelist:
        image_name = file.split('.')[0]
    

        imagepath= imagefoldername + '/' + image_name + '.' + settings['train']['img_ext']
        img=imread(imagepath)
        train_input_name.append(image_name);
        if patch_size != 0:
            img1=imresize(img,(patch_size,patch_size,3));
        else:
            img1=img;
        I2=img1.flatten()
        test_input[count,:]=I2

#       imagepath= maskfoldername + '/' + image_name + '.' + settings['train']['lbl_ext']
 #      img=imread(imagepath)
  #     if patch_size != 0:
   #        img1=imresize(img,(patch_size,patch_size,1));
    #   else:
     #      img1=img;
      # print(I2.shape)
       #I2=img1.flatten()
       #train_ouput[count,:]=I2


        #ount=count+1
    return test_input, train_input_name
    

    
# Preprocess the image and flatten     
Xa,train_input_name =downsample_flatten1(inputimagedata,num_images,3,patch_size,img_size)
#Y =downsample_flatten2(maskedimagedata,3405,1) 

# Manually split the data for training and testing 
num_test_img = int(np.floor(num_images * settings['train']['percent_test_img']/100));
#x_train=Xa[num_test_img:,:]
test_input_name=train_input_name;
#x_test=Xa[:num_test_img,:]
#y_train=Y[num_test_img:,:]
#y_test=Y[:num_test_img,:]
#y_test=Y
x_test=Xa
patch_size = settings['patch_size'];
 
# Input rows, cols, no of channels
if patch_size != 0: 
    img_rows=patch_size;
    img_cols=patch_size;
else:
    img_rows=img_size;
    img_cols=img_size;
    
img_channels=3
img_channels_Y=1

# Normalize the images
#x_train_label = y_train.astype('float32') / 255.0
#x_test_label = y_test.astype('float32') / 255.0
#x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# restore the original shape from the flatten vector
#x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,img_channels)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,img_channels)
#x_train_label=x_train_label.reshape(x_train_label.shape[0],img_rows,img_cols,img_channels_Y)
#x_test_label=x_test_label.reshape(x_test_label.shape[0],img_rows,img_cols,img_channels_Y)

#print(x_train.shape)
print(x_test.shape)

# visualize the input and labeled images for sanity check
'''
n = 5
plt.figure(figsize=(10, 4.5))
for i in range(n):
    # plot labeled image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_label[i].reshape(s, s))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    if i == n/2:
        ax.set_title('Labeled Images')

    # plot original image 
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test[i].reshape(s, s,3))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    if i == n/2:
        ax.set_title('Orinigal Images')
'''
# Define the deep autoencoder model

loss_func = settings['train']['loss'];
optimizer_fun = settings['train']['optimizer'];
autoencoder = createModel(img_rows, img_cols, img_channels);
autoencoder.compile(optimizer=optimizer_fun, loss=loss_func,metrics=['accuracy'])


autoencoder.load_weights(cwd+'/'+name_model+'/'+name_model+'.h5');

print('**********')


## test the model (basically for visualization, testing is already done above)
import os

file_path = cwd +'/'+name_model;


final_directory = os.path.join(file_path, name_model+'_result')
save_path = file_path + '/'+name_model+'_result'
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

plt.figure(figsize=(10, 7))
images = autoencoder.predict(x_test)
print(num_images)
for i in range(images.shape[0]):
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(x_test[i].reshape(img_rows, img_cols,3))
    plt.subplot(122)
    plt.imshow(images[i].reshape(img_rows, img_cols))
    
#    plt.subplot(133)
#    plt.imshow(x_test_label[i].reshape(img_rows, img_cols))
    
    fig.savefig(save_path + '/' + test_input_name[i] + '.png')

