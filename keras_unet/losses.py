"""
This file works to define the loss functions available to use during the training / tuning process. 
While some different loss functions are defined, the final loss function chosen was the bce_dice_loss function which combines binary cross entropy and dice coefficient loss in an effort to better segment datasets where some images have blank masks. 
"""

from keras import backend as K
from keras.losses import binary_crossentropy
#from metrics import dice_coef

"""
The dice coefficient is a segmentation metric that helps analyze the ability for the image to evaluate how well it calculated the true values divided by all the real true values or 2*intersection (or TP) divided by 2* union or (2*TP + FN + FP)
Based on the wikipedia post and medium article https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=Simply%20put%2C%20the%20Dice%20Coefficient,of%20pixels%20in%20both%20images.
"""
def dice_coef(truth, prediction,smooth=0):
    true_f = K.flatten(truth) #flatten the true matrix for calculation
    #line taken from internet (forgotten source, to speed up loss function, threshold_binarize was too slow in computation)
    pred_f = K.cast(K.greater(K.flatten(prediction), 0.5), 'float32') #flatten and binarize the predicted matrix
    intersection = true_f * pred_f #create calculation for which values are intersecting
    score = 2. * K.sum(intersection) / (K.sum(true_f) + K.sum(pred_f)) #use final formula to divide intersecting true values by all values
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
