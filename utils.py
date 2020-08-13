"""
This utils script is a script for loading in images and plotting the figures. This differs from the other utils function which revolves around work with the model. This function works to create utility functions that help with loading in the data, for evaluation and training, and plotting the figures for evaluation. 
"""

import os
import sys 
import random
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import tqdm

#load all the LGG brain MRI images into the model 
def load_images(main_path):
    #create empty arrays to contain the images 
    image_path,mask_path=[],[]
    #list out the folders available from the main path
    folders=os.listdir(main_path)
    #loop through the folders joining the mask images with the mask array and the MR images with the image array 
    for folder in folders:
        tmp_path=os.path.join(main_path,folder)
        for file in os.listdir(tmp_path):
            if 'mask' in file.split('.')[0].split('_'):
                mask_path.append(os.path.join(tmp_path,file))
            else:
                image_path.append(os.path.join(tmp_path,file))
                
    return image_path, mask_path

#Skin Images were saved in a different format 
#This function allows the loading of skin images into the file 
#function to load the skin images 
def read_in_skin_images(skin_path,msk_path,img_path):
    Images = []
    Masks = []
  
    for img in img_path:
        temp_img = skin_path+'/image/' +img
        temp_img = cv2.imread(temp_img)
        Images.append(temp_img)

    for msk in msk_path:
        temp_msk = skin_path+ '/mask/'+msk
        temp_msk = cv2.imread(temp_msk)
        Masks.append(temp_msk)

    return Images, Masks




#Lung images were saved in a different format
#This function allows the loading of lung images into the file 
#function to load in lung images 
def read_in_lung_images(lung_path,msk_path,img_path):
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

#for the first 9 images in the array plot a 3 by 3 matrix of their images overlayed by mask
def plot_figures(img_path, msk_path):
    rows,cols=3,3
    fig=plt.figure(figsize=(10,10))
    #loop through the first 9 image mask pairs 
    for i in range(1,rows*cols+1):
        #add a square in the area you want
        fig.add_subplot(rows,cols,i)
        #load in images and masks 
        img=img_path[i]
        msk=msk_path[i]
        #convert the value to 
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(msk,alpha=0.4)
    plt.show()

    
#Return boolean of whether or not all images have masks    
def image_mask_check(image_path, mask_path):
    img_wm=[]
    for img_p in image_path:
        img_p=img_p.split('.')
        img_p[0]=img_p[0]+'_mask'
        img_p='.'.join(img_p)
        if img_p not in mask_path:
            img_wm.append(img_p)

    del mask_path
    mask_path=[]
    for img_p in image_path:
        img_p=img_p.split('.')
        img_p[0]=img_p[0]+'_mask'
        img_p='.'.join(img_p)
        mask_path.append(img_p)
	
    if len(img_wm)==0:
        return True
    else:
        return False

#Creates array of images read in from image path
def read_in_MR_images(msk_path,img_path):
    #create empty array of images and masks 
    Images = []
    Masks = []
    #for each respective array read in the images 
    for img in img_path:
        temp_img = cv2.imread(img)
        Images.append(temp_img)

    for msk in msk_path:
        temp_msk = cv2.imread(msk)
        Masks.append(temp_msk)

    return Images, Masks
