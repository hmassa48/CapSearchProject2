"""
This file works to define some different metrics that are used when evaluating the images. 
"""

import tensorflow as tf
import numpy as np
from keras import backend as K

#function that allows the thresholding of mask images 
def threshold_binarize(mask, threshold=0.5):
    above_thresh = tf.greater_equal(mask, tf.constant(threshold)) #find values above threshold
    #values that are not above the threshold mark as 0 in thresholded mask, otherwise 1
    thresh_mask = tf.where(above_thresh, mask=tf.ones_like(mask), thresh_mask=tf.zeros_like(mask)) 
    return thresh_mask

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


""" 
IOU metric calculates the intersection over the Union. This helps analyze the segmentation. 
This was calculated similar to Dice Coefficient using the same medium article 
https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=Simply%20put%2C%20the%20Dice%20Coefficient,of%20pixels%20in%20both%20images.
changes were made to the overall code to make it easier to use and understand
"""
def iou(truth, prediction, smooth=1.):
    #flatten both arrays for calculation
    truth_f = K.flatten(truth)
    pred_f = K.cast(prediction, 'float32') 
    pred_f = K.flatten(pred_f)
    #calculate the intersection of the values 
    intersection = K.sum(truth_f * pred_f)
    return (intersection + smooth) / (K.sum(truth_f) + K.sum(pred_f) - intersection + smooth)

#evaluate IOU thresholded to compare to actual IOU value 
def iou_thresholded(truth, prediction, threshold=0.5, smooth=1.):
    #binarize the prediction image
    thres_pred = threshold_binarize(prediction, threshold)
    return iou(truth, thres_pred, smooth)

""" 
Pixel Accuracy, Precision, Recall Calculated using the basis of the blog post below 
https://www.jeremyjordan.me/evaluating-image-segmentation-models/

"""

#accuracy is defined as the correct/total 
def pixelwise_accuracy(truth, prediction):
    #number of images to loop through
    num_images = truth.shape[0]
    #correct and total for final division
    sum_correct = 0
    sum_total = 0
    
    for i in range(0,num_images):
        #evaluate current mask 
        eval_prediction = prediction[i,:,:]
        eval_test = truth[i,:,:]
        #add up similarities and total values 
        sum_total += eval_test.shape[0] *eval_test.shape[1]
        for w in range(0,eval_test.shape[0]):
            for h in range(0,eval_test.shape[1]):
                if (eval_prediction[w,i] == eval_test[w,i]):
                    sum_correct += 1
    return sum_correct/sum_total

#threshold the pixelwise accuracy value using two functions we already have 
def thresholded_pixelwise_accuracy(truth, prediction):
    new_preds = threshold_binarize(prediction,0.5)
    return pixelwise_accuracy(truth, new_preds)

#precision is known as the TP/(TP+FP)
#this shows the purity 

def thresholded_precision(truth,prediction):
    #binarize threshold
    new_preds = threshold_binarize(prediction, 0.5)
    #starting loop and shape variables 
    num_images = truth.shape[0]
    num_TP = 0
    num_predictedPositives = 0 
    
    #loop through all images 
    for i in range(0,num_images):
        #evaluate current mask 
        eval_prediction = new_preds[i,:,:]
        eval_test = truth[i,:,:]
        #find total positive guesses 
        for w in range(0,eval_test.shape[0]):
            for h in range(0,eval_test.shape[1]):
                if (eval_prediction[w,h] == 1):
                    num_predictedPositives +=1
                    if (eval_test[w,h] == 1):
                        num_TP +=1 
    
    if (num_predictedPositives ==0):
        return 0 
    else:
        return num_TP/num_predictedPositives 
    
#TP / TP + FN 
#recall shows the completeness 
def thresholded_recall(truth,prediction):
    #binarize threshold
    new_preds = threshold_binarize(prediction, 0.5)
    #starting loop and shape variables 
    num_images = truth.shape[0]
    num_TP = 0
    num_allPositives = 0 
    
    #loop through all images 
    for i in range(0,num_images):
        #evaluate current mask 
        eval_prediction = new_preds[i,:,:]
        eval_test = truth[i,:,:]
        #find total positive guesses 
        for w in range(0,eval_test.shape[0]):
            for h in range(0,eval_test.shape[1]):
                if (eval_test[w,h] == 1):
                    num_allPositives +=1
                    if (eval_prediction[w,h] == 1):
                        num_TP +=1 
    
    if (num_allPositives ==0):
        return 0 
    else:
        return num_TP/num_allPositives 

def mse(mask_test,preds):
    return (np.square(preds - mask_test)).mean(axis=None)

def mae(mask_test,preds):
    return np.abs(preds-mask_test).mean(axis=None)

