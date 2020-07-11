import os
import sys 
import random
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import tqdm

def load_images(main_path):
    image_path,mask_path=[],[]
    
    folders=os.listdir(main_path)
    folders.remove('data.csv')
    folders.remove('README.md')
    for folder in folders:
        tmp_path=os.path.join(main_path,folder)
        for file in os.listdir(tmp_path):
            if 'mask' in file.split('.')[0].split('_'):
                mask_path.append(os.path.join(tmp_path,file))
            else:
                image_path.append(os.path.join(tmp_path,file))
                
    return image_path, mask_path

def load_lung_images(image_path,mask_path):
    #get just image name
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    
    masks = [file.split(".png")[0] for file in masks]
    image_file_name = [file.split("_mask")[0] for file in masks]
    
    #files with the same name
    same_name = set(os.listdir(image_path)) & set(os.listdir(mask_path))
    
    #files with masks
    other_masks = [i for i in masks if "mask" in i]
    final_masks = list(same_name) + [x + '.png' for x in other_masks]
    final_images = [x.split('_mask')[0] for x in other_masks]
    final_images = list(same_name) + [x + '.png' for x in final_images]
    
    #sort
    final_masks.sort()
    final_images.sort()
    #get full path
    final_images = [image_path + x for x in final_images]
    final_masks = [mask_path + x for x in final_masks]
    
    return final_images, final_masks


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

def plot_figures(img_path, msk_path):
    rows,cols=3,3
    fig=plt.figure(figsize=(10,10))
    for i in range(1,rows*cols+1):
        fig.add_subplot(rows,cols,i)
        i_path=img_path[i]
        m_path=msk_path[i]
        img=cv2.imread(i_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        msk=cv2.imread(m_path)
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