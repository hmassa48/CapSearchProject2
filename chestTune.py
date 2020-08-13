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


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


NUM_EPOCH_SEARCH = 50

def read_in_images(lung_path,msk_path,img_path):
    Images = []
    Masks = []
  
    for img in img_path:
        temp_img = lung_path+'/2d_images/' +img
        temp_img = cv2.imread(temp_img)
        Images.append(temp_img)

    for msk in msk_path:
        temp_msk = lung_path+ '/2d_masks/'+msk
        temp_msk = cv2.imread(temp_msk)
        Masks.append(temp_msk)

    return Images, Masks



#load dataset
lung_path = 'lung-masks'

img_path = lung_path + '/2d_images/'
msk_path = lung_path + '/2d_masks/'

imgs = os.listdir(img_path)
msks = os.listdir(msk_path)

msks = sorted(msks)
imgs = sorted(imgs)

images,masks = read_in_images(lung_path,msks,imgs)


        
for i in range(0,len(masks)):
    m = masks[i]
    m = m[:,:,0]
    m.reshape(m.shape[0],m.shape[1])
    m = cv2.resize(m,(256,256))
    masks[i] = m
    #images
    im = images[i]
    images[i] = cv2.resize(im,(256,256))

#make Arrays 
images = np.asarray(images)
masks = np.asarray(masks)
masks = masks / 255
masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1)

#split into validations
img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)


train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    img_overall_train, mask_overall_train,
    batch_size=16)

val_generator = train_datagen.flow(
    img_test, mask_test)

hp = HyperParameters()
#fix algorithmic hyperparameters
hp.Fixed('learning_rate', value = 0.001)
hp.Fixed('optimizer_name', value = 'adam')
hp.Fixed('kernel', value = 3)

hypermodel = HyperBasicUNet(input_shape = (256,256,3), classes = 1)

tuner_hb = Hyperband(
            hypermodel,
            max_epochs=200,
            objective='val_loss',
            metrics = [iou,iou_thresholded,'mse'],
            distribution_strategy=tf.distribute.MirroredStrategy(),
            seed=42,
            hyperband_iterations = 3
        )

tuner_hb.search_space_summary()


tuner_hb.search(train_generator,epochs = 500,verbose = 1,validation_data = (img_test,mask_test))

best_model = tuner_hb.get_best_models(1)[0]

best_hyperparameters = tuner_hb.get_best_hyperparameters(1)[0]
print(tuner_hb.get_best_hyperparameters(1))

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
tuner_hb.results_summary()
