"""
This file is an establishment of some utility functions for different data augmentation techniques. Depending on the medical imaging data used, 
these augmentation techniques might be useful. 
"""


import numpy as np
from skimage.exposure import rescale_intensity
import random
from skimage.transform import rotate, AffineTransform, warp

#used in the MR images (as well as the breast cancer cell images) for color intensity normalization

"""
Based off the research: https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-015-0064-y#:~:text=Intensity%20normalization%20is%20an%20important,resonance%20image%20(MRI)%20analysis.&text=This%20intensity%20variation%20will%20greatly,segmentation%2C%20and%20tissue%20volume%20measurement.
This is a simple way to do the histogram based normal curve idea of normalization of the MR images that the work discussed mathematically.
"""
def normalize_MRIvolume(MRI_volume):
    #calculate intensity percentiles 
    #using 10 and 99, we want to keep the top information but can take less of lower percentile 
    p10 = np.percentile(MRI_volume, 10)
    p99 = np.percentile(MRI_volume, 99)
    #rescale the intensity to be within these percentiles 
    MRI_volume = rescale_intensity(MRI_volume, in_range=(p10, p99))
    #normalize with this new rescaled intensity
    m = np.mean(MRI_volume, axis=(0, 1, 2))
    s = np.std(MRI_volume, axis=(0, 1, 2))
    #return normalized volume
    MRI_volume = (MRI_volume - m) / s
    return MRI_volume


"""
Set Up for Pre-processing each of the image and mask 
"""
def preprocess_img_msks(images,masks,target_size, num_classes, isMR = False, isGray = False ):
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0]
        m.reshape(m.shape[0],m.shape[1])
        m = cv2.resize(m,target_size)
        masks[i] = m
        #images
        im = images[i]
        im = cv2.resize(im,target_size)
        if isMR:
            img = normalize_MRIvolume(im)
        images[i] = im
    
    #convert to numpy
    images = np.asarray(images, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    
    #divide by the largest amount to normalize 
    masks = masks / masks.max()
    if isGray:
        images = images / images.max()

    masks = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2], num_classes)

    if isGray:
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    return images,masks 




"""
Simple Data Augmentations 
Not used in this work explicitly but prepared in case they would be used to create more data and more robust algorithms. 
This work decided to evaluate the U-Net on data that did not create more synthetic images. 
"""
def rotate_clockwise(image):
    rot_ang = random.randint(0,100)
    return rotate(image,-rot_ang)

def rotate_counterclockwise(image):
    rot_ang = random.randint(0,100)
    return rotate(image,rot_ang)

def flip_image(image):
    return np.fliplr(image)


