import tensorflow as tf
from keras import backend as K

#metric for the intersection over union bound
def iou(y_true, y_pred, smooth=1.):
    #flatten out the array of evaluation images 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #calculate the intersection between images 
    intersection = K.sum(y_true_f * y_pred_f)
    #return the intersection over the boundary of images 
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

#create a binary threshold for more accurate IOU (thresholded metric)
def threshold_binarize(x, threshold=0.5):
    #calculate if a factor is greater than or equal to the binary constant 
    ge = tf.greater_equal(x, tf.constant(threshold))
    #anywhre it is greater than mark as a one, anywhere it is less mark as 0 
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

#create the metric for the thresholded IOU
def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    #binarize the prediced image 
    y_pred = threshold_binarize(y_pred, threshold)
    #perform the same IOU technique from above 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

#create a metric for th pixel-wise accuracy of a function 
def pixelwise_accuracy(truth, prediction):
    num_images = truth.shape[0]
    sum_n_ii = 0
    sum_t_i = 0
    
    for i in range(0, num_images):
        curr_eval_mask = truth[i, :, :]
        curr_gt_mask = prediction[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
        
    if (sum_t_i == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n_ii / sum_t_i

    return pixel_accuracy
