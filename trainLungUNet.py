"""
This is a script that train the Lung U-Net model. Used to train both the traditional U-Net and tuned parameter U-Net, by changing custom U-Net parameters in script. 
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
import cv2
from keras_unet.models import custom_unet

from keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import bce_dice_loss



import numpy as np 



#Upload and prepare data 
def main():
    #trainiing settings 

    #Multi GPU set up
    strategy = tf.distribute.MirroredStrategy()    


    #load dataset
    lung_path = 'Data/lung-masks'

    img_path = lung_path + '/2d_images/'
    msk_path = lung_path + '/2d_masks/'

    #list all images and masks 
    imgs = os.listdir(img_path)
    msks = os.listdir(msk_path)
    #sort images and masks
    msks = sorted(msks)
    imgs = sorted(imgs)

    #read in images and masks 
    images,masks = read_in_lung_images(lung_path,msks,imgs)

    #set up basic preprocessing 
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0] #binarize the masks
        m.reshape(m.shape[0],m.shape[1]) #reshape binary masks 
        m = cv2.resize(m,(256,256)) #resize the masks due to memory constraints 
        masks[i] = m
        #images
        im = images[i]
        images[i] = cv2.resize(im,(256,256)) #resize the images due to memory constraints 

    #make Arrays 
    images = np.asarray(images)
    masks = np.asarray(masks)
    masks = masks / masks.max() #normalize the masks 
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1) #reshape the masks to binary set up 
    

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

    #standardize steps per epoch based on the dataset 
    STEPS_PER_EPOCH = len(img_train) // 16
    #Get U Net 


    input_shape = img_train[0].shape #get input shape for the U-Net model

    #use multiGPU training strategy	
    with strategy.scope():
	#change parameters based on model you want to train 
	#set up custom u net model 
        model = custom_unet(
        input_shape,
        filters=64,
        use_batch_norm=False,
        use_dropout_on_upsampling = False,
        dropout=0.55,
        activation = 'relu',
        dropout_change_per_layer=0.00,
        num_layers=4,
        decoder_type = 'simple'
    )

    

    ##Compile and Train
        #save model name based on model you are training and want to evaluate
        model_filename = 'traditional_600_LR0001_BASIC_LUNG_UNET.h5'
        callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_loss', 
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
