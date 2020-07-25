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

#function to load in lung images 
def read_in_images(lung_path,msk_path,img_path):
    #create empty array to return with the lung images 
    Images = []
    Masks = []
  
    #loop through lung and mask images 
    #create the path to those images and read in the values with cv2
    #append the image to the respective array 
    for img in img_path:
        temp_img = lung_path+'/2d_images/' +img
        temp_img = cv2.imread(temp_img)
        Images.append(temp_img)

    for msk in msk_path:
        temp_msk = lung_path+ '/2d_masks/'+msk
        temp_msk = cv2.imread(temp_msk)
        Masks.append(temp_msk)

    return Images, Masks



def main():
    #trainiing settings 

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy() 
    
    #load dataset
    lung_path = 'lung-masks'

    #create the image paths
    img_path = lung_path + '/2d_images/'
    msk_path = lung_path + '/2d_masks/'
    
    #list out both of the directories using os 
    imgs = os.listdir(img_path)
    msks = os.listdir(msk_path)
    
    #sort the images and masks 
    msks = sorted(msks)
    imgs = sorted(imgs)

    #use the read in lung image function to read in the images 
    images,masks = read_in_images(lung_path,msks,imgs)
    
    #reshape mask image to (256,256) -- binarize
    #remove the third component of the mask image 
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0]
        m.reshape(m.shape[0],m.shape[1])
        masks[i] = m

    #make images and masks into numpy arrays 
    images = np.asarray(images)
    masks = np.asarray(masks)
    #normalize the mask values 
    masks = masks / 255
    #reshape the final mask for training in the neural network 
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1)

    #split the data using sklearn 
    img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)
    img_train, img_val, mask_train, mask_val = train_test_split(img_overall_train, mask_overall_train, test_size = 0.166667, random_state = 32)
   
    #create a data generator to hold the information for training 
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        img_train, mask_train,
        batch_size=16)

    #steps per epoch is the length of total training divided by the batch size 
    STEPS_PER_EPOCH = len(img_train) // 16
    #Get U Net 

    #get model input shape 
    input_shape = img_train[0].shape
    
    #set up the keras code to work with multiple GPUS 
    #train the model within the scope that will allow it to 
    with strategy.scope():
        model = custom_unet(
        input_shape,
        filters=64,
        use_batch_norm=False,
        dropout=0.55,
        dropout_change_per_layer=0.0,
        num_layers=4,
        decoder_type = 'simple'
    )

    
    ##Compile and Train

    # ... then onto training the model that uses much more code from other sections and is replicated elsewhere 
