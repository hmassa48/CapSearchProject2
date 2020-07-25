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

#Lung images were saved in a different format
#This function allows the loading of lung images into the file 
def load_lung_images(image_path,mask_path):
    #get just image name from the paths 
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    
    #split the mask name so that it will match the image name
    masks = [file.split(".png")[0] for file in masks]
    image_file_name = [file.split("_mask")[0] for file in masks]
    
    #files with the same name
    #set all images and masks with the same name together 
    same_name = set(os.listdir(image_path)) & set(os.listdir(mask_path))
    
    #other mask files were named something completely different 
    other_masks = [i for i in masks if "mask" in i]
    final_masks = list(same_name) + [x + '.png' for x in other_masks]
    final_images = [x.split('_mask')[0] for x in other_masks]
    final_images = list(same_name) + [x + '.png' for x in final_images]
    
    #sort the images so that they will line up 
    final_masks.sort()
    final_images.sort()
    #get full path
    final_images = [image_path + x for x in final_images]
    final_masks = [mask_path + x for x in final_masks]
    
    return final_images, final_masks

#checks to see if every MR and Mask image have a pair 
def image_mask_check(image_path, mask_path):
    #Return boolean of whether or not all images have masks 
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

def read_in_images(msk_path,img_path):
    Images = []
    Masks = []
  
    for img in img_path:
        temp_img = cv2.imread(img)
        Images.append(temp_img)

    for msk in msk_path:
        temp_msk = cv2.imread(msk)
        Masks.append(temp_msk)

    return Images, Masks
