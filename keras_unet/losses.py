"""
This file works to define the loss functions available to use during the training / tuning process. 
While some different loss functions are defined, the final loss function chosen was the bce_dice_loss function which combines binary cross entropy and dice coefficient loss in an effort to better segment datasets where some images have blank masks. 
"""

from keras import backend as K
from keras.losses import binary_crossentropy

#based on the wikipedia article and medium post on how to calculate the dice coefficient 
#the dice coefficient helps to take 2 *intersection (or TP values ) divided by 2* the union (or 2*TP + FN + FP) values 
def dice_coef(y_true, y_pred,smooth=0):
    y_true_f = K.flatten(y_true) #flatten the true matrix for calculation
    y_pred = K.cast(y_pred, 'float32') #make sure prediction values are float to make true
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32') #flatten and binarize the predicted matrix
    intersection = y_true_f * y_pred_f #create calculation for which values are intersecting
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f)) #use final formula to divide intersecting true values by all values
    return score

#This function creates loss based off of the dice coefficient 
def dice_loss(y_true, y_pred,smooth = 1):
    dc = dice_coef(y_true,y_pred,smooth)
    return 1. - dc

#This is the function used in work 
#It is a combination of binary cross entropy and the dice loss function from above 
#This was created to better evaluate the segmentation problem of returning a blank mask image with high accuracy
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
