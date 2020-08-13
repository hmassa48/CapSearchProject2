"""
Used to train the LGG Brain Segmentation U-Net model. This was used to train the traditional U-Net and tuned parameter U-Net by changing the parameters. 
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.engine import hypermodel
from keras.callbacks import ModelCheckpoint
from utils import *
import tqdm
from sklearn.model_selection import train_test_split

from keras_unet.models import custom_unet
from keras_unet.model_utils import normalize_MRIvolume

from keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import bce_dice_loss

import numpy as np 



#Upload and prepare data 
def main():
    #trainiing settings 

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()   

    #load dataset
    image_paths,mask_paths = load_images("Data/lgg-mri-segmentation/kaggle_3m/")

    #sort images and masks
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    #read in images and masks 
    if image_mask_check(image_paths, mask_paths):
        masks,images = read_in_MR_images(image_paths,mask_paths)
    
    #basic pre-processing with MR additives
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0] #binarize the masks 
        m.reshape(m.shape[0],m.shape[1]) #set up binarize shape for masks
        masks[i] = m
        #MRI Values
        im = images[i]
        images[i] = normalize_MRIvolume(im) #normalize the MR images with histogram normalization

    #make Arrays 
    images = np.asarray(images)
    masks = np.asarray(masks)
    masks = masks / masks.max() #normalize masks
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1) #set up binarize shape for masks


    #split the data
    img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)
    img_train, img_val, mask_train, mask_val = train_test_split(img_overall_train, mask_overall_train, test_size = 0.166667, random_state = 32)
   
    #data generator 
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        img_train, mask_train,
        batch_size=16)

    val_generator = train_datagen.flow(
        img_val, mask_val)


    #standardize steps per epoch based on image dataset and batch size 
    STEPS_PER_EPOCH = len(img_train) // 16
   # STEPS_PER_EPOCH = 250
    
    
    #Get U Net 

    input_shape = img_train[0].shape #input shape for the U-Net

    #multiGPU processing
    with strategy.scope():
        #train custom U-Net model based on parameters you want to train
        model = custom_unet(
        input_shape,
        filters=64,
        use_batch_norm=False,
        dropout=0.55,
        dropout_change_per_layer=0.00,
        num_layers=4,
        decoder_type = 'simple',
        use_dropout_on_upsampling =False,
        activation = 'relu'
    )

    

    ##Compile and Train
        #save model name for evaluation
        model_filename = 'Basic_LGG_aug_UNET_01LR.h5'
        callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='loss', 
        save_best_only=True,
    )
        opt = keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(
            optimizer=opt, 
        #optimizer=SGD(lr=0.01, momentum=0.99),
            loss= bce_dice_loss,
            metrics=['mse',iou, iou_thresholded],
    )


        history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=600,
        validation_data=(img_val, mask_val),
        callbacks=[callback_checkpoint]
)

    # serialize model to JSON
    model_json = model.to_json()
    with open("bcdloss_LGG_basic_UNetModel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("LGG_basic_UNetmodel.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
