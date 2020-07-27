"""
This file works to define some different metrics that are used when evaluating the images. 
"""

import tensorflow as tf
import numpy as np
from keras import backend as K


def iou(truth, prediction, smooth=1.):
    truth_f = K.flatten(truth)
    pred_f = K.flatten(prediction)
    intersection = K.sum(truth_f * pred_f)
    return (intersection + smooth) / (K.sum(truth_f) + K.sum(pred_f) - intersection + smooth)

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

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

