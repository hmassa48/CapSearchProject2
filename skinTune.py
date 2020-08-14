"""
Script to find the best architectures for the Skin Lesion Dataset
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
import cv2
from utils import *
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import bce_dice_loss
from keras_unet.models import HyperBasicUNet
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#load dataset
skin_path = 'Data/skin_lesions'

#paths to images and masks within the overall skin path
img_path = skin_path + '/image/'
msk_path = skin_path + '/mask/'

#list possible images and masks
imgs = os.listdir(img_path)
msks = os.listdir(msk_path)

#sort images and masks to correlate same images with masks
msks = sorted(msks)
imgs = sorted(imgs)

#read in the skin images and masks
images,masks = read_in_skin_images(skin_path,msks,imgs)

#preprocess the skin images 
        
for i in range(0,len(masks)):
    m = masks[i]
    m = m[:,:,0] #binarize the mask
    m.reshape(m.shape[0],m.shape[1]) #reshape the binary mask
    m = cv2.resize(m,(256,256)) #resize the masks to a standard size for memory constraints
    masks[i] = m
    #images
    im = images[i]
    images[i] = cv2.resize(im,(256,256)) #resize the images to a standard size for memory constraints


#convert the images to arrays
images = np.asarray(images)
masks = np.asarray(masks)
masks = masks / masks.max() #normalize the mask images 
masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1) #save mask images in a binary format


#split into validations
img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)

#data rotation values 
data_aug_dict = dict(
       # rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

#data generator 
train_datagen = ImageDataGenerator(data_aug_dict)

train_generator = train_datagen.flow(
    img_overall_train, mask_overall_train,
    batch_size=16)

val_generator = train_datagen.flow(
    img_test, mask_test)

#load the keras-tuner U-Net architecture 
hypermodel = HyperBasicUNet(input_shape = (256,256,3), classes = 1)

#create the hyperband search strategy for the keras-tuner 
#different values chosen based on memory constraints
tuner_hb = Hyperband(
            hypermodel,
            max_epochs=200,
            objective='val_loss',
            metrics = [iou,iou_thresholded,'mse'],
            distribution_strategy=tf.distribute.MirroredStrategy(),
            seed=42,
            hyperband_iterations = 3
        )

#print total search space for reference
tuner_hb.search_space_summary()

#search through the search space
tuner_hb.search(train_generator,epochs = 500,verbose = 1,validation_data = (img_test,mask_test))

#save best models and hyperparameters
best_model = tuner_hb.get_best_models(1)[0]

best_hyperparameters = tuner_hb.get_best_hyperparameters(1)[0]
print(tuner_hb.get_best_hyperparameters(1))

#save best models and weights
model_json = best_model.to_json()
with open("hp_bce_all_basicUNet_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("hp_bce_all_tune_basicUNet_tuner_model.h5")

with open('best_LGG_basicUNet_Param.txt', 'w') as f:
    print(best_hyperparameters, file=f)

print("Saved model to disk")
print(best_hyperparameters)
#show top 10 architectures 
tuner_hb.results_summary()
