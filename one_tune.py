"""
Script used to tune the LGG Brain MR dataset through the use of the Keras-Tuner and the defined HyperBasicUNet class
"""

import kerastuner as kt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
	concatenate,
        multiply,
        add,
	Activation)
from kerastuner.engine import hypermodel
from kerastuner.tuners import Hyperband
from utils import *
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import bce_dice_loss
from keras_unet.model_utils import normalize_MRIvolume
from keras_unet.models import HyperBasicUNet

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#load in image dataset 
image_paths,mask_paths = load_images("Data/lgg-mri-segmentation/kaggle_3m/")

if image_mask_check(image_paths, mask_paths):
    masks,images = read_in_MR_images(image_paths,mask_paths)

#preprocess the MR images 
for i in range(0,len(masks)):
    m = masks[i]
    m = m[:,:,0] #binarize end of mask image
    m.reshape(m.shape[0],m.shape[1]) #reshape mask to the correct size 
    masks[i] = m
    #MRI Values
    im = images[i]
    images[i] = normalize_MRIvolume(im) #normalize MR value with histogram normalization

#convert images and masks to arrays 
images = np.asarray(images) 
masks = np.asarray(masks)
masks = masks/masks.max() #normalize the mask images 
masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1) #reshape masks into binary mask shape (:,:,:,1)

#split into validations
img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)

#create image data generators to define batch size 
train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    img_overall_train, mask_overall_train,
    batch_size=16)

val_generator = train_datagen.flow(
    img_test, mask_test)

#load in the defined hyper U-Net architecture 
hypermodel = HyperBasicUNet(input_shape = (256,256,3), classes = 1)

#create the hyperband search strategy
tuner_hb = Hyperband(
            hypermodel,
            max_epochs=200,
            objective='val_loss',
            metrics = [iou,iou_thresholded,'mse'],
            distribution_strategy=tf.distribute.MirroredStrategy(),
            seed=42,
            hyperband_iterations = 3
        )

#print out the search space summary
tuner_hb.search_space_summary()

#search the search space
tuner_hb.search(train_generator,epochs = 500,verbose = 1,validation_data = (img_test,mask_test))

best_model = tuner_hb.get_best_models(1)[0]

best_hyperparameters = tuner_hb.get_best_hyperparameters(1)[0]


#
model_json = best_model.to_json()
with open("hp_bce_all_basicUNet_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("hp_bce_all_tune_basicUNet_tuner_model.h5")

with open('best_LGG_basicUNet_Param.txt', 'w') as f:
    print(best_hyperparameters, file=f)


print("Saved model to disk")
print(best_hyperparameters)
