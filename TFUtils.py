#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:11:10 2018

@author: Niny
"""

import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import scipy
from PIL import Image

import os

IMAGE_SIZE=256

#recordPath = '/media/xenia/Niny_Q/ISELS2018/Unet/'
#recordPath = 'Q:/Codes/ISLES2018/ISLES/'
recordPath = os.getcwd()+'/'


#gan gt to scorcmap
def convert_to_scaling(fk_batch, num_classes, label_batch, tau=0.9):
    lab_hot = tf.squeeze(tf.one_hot(label_batch, num_classes, dtype=tf.float32), axis=3)

    # fk_batch = tf.nn.softmax(fk_batch, dim=-1)
    fk_batch_max = tf.reduce_max(fk_batch, axis=3, keep_dims=True)
    fk_batch_max = tf.maximum(fk_batch_max, tf.fill(tf.shape(fk_batch_max), tau))
    fk_batch_maxs = tf.concat([fk_batch_max for i in range(num_classes)], axis=3)
    gt_batch = tf.where(tf.equal(lab_hot, 1.), fk_batch_maxs, fk_batch)
    y_il = 1. - fk_batch_maxs
    s_il = 1. - fk_batch
    y_ic = tf.multiply(fk_batch, tf.div(y_il, s_il))
    gt_batch = tf.where(tf.equal(lab_hot, 0.), y_ic, gt_batch)
    sums = tf.reduce_sum(gt_batch, axis=3)
    temp = tf.expand_dims((sums - tf.ones_like(sums, dtype=tf.float32)) / num_classes, axis=3)
    gt_batch = gt_batch - tf.concat([temp for i in range(num_classes)], axis=3)

    return gt_batch



def soft_dice_loss(logits, ground_truth):
    #probabilities = tf.sigmoid(logits)
    probabilities = tf.softmax(logits)
    interception_volume = tf.reduce_sum(probabilities * ground_truth)
    return - 2 * interception_volume + tf.constant(smooth) / (tf.norm(ground_truth, ord=1) + tf.norm(probabilities, ord=1) + tf.constant(smooth))

def cross_entropy_loss(logits, ground_truth):
    # flatten
    logits = tf.reshape(logits, (-1,))
    ground_truth = tf.reshape(ground_truth, (-1,))
    # calculate reduce mean sigmoid cross entropy (even though all images are different size there's no problem)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ground_truth), name='loss')

def mixed_loss(logits, ground_truth):
    return cross_entropy_loss(logits, ground_truth) + soft_dice_loss(logits, ground_truth)


def focal_loss(prediction, groundtruth, weights=None, alpha=0.25, gamma=2):
    """
    Compute the focal loss
    
    FL = -alpha*(1-p)^gamma*log(p)
    
    alpha=0.25, gamma=2, p=sigmoid(x) z=label
    
    
    return a tensor of length batch_size of same type as logits with softmax focal loss
    """
    p = tf.nn.sigmoid(prediction)
    pred_pt = tf.where(tf.equal(groundtruth,1),p, 1.-p)
    
    epsilon=1e-8
    
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(groundtruth, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(groundtruth, 1), alpha_t, 1-alpha_t)
    
    f_loss = tf.reduce_mean(-alpha_t * tf.pow(1.-pred_pt, gamma) * tf.log(pred_pt+epsilon))
    
    return f_loss


#Save images
def save_imgs(test_num, label_batch, pred_batch, itr):
    
    savepath = recordPath+'imgs/'+'results'+str(itr)+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    label_img_mat = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    pred_img_mat = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    for i in range (IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if label_batch[i, j] == 0:
                label_img_mat[i, j, 0] = label_img_mat[i, j, 1] =  label_img_mat[i, j, 2] = 128  # backgroud Gray
            if label_batch[i, j] == 1:
                label_img_mat[i, j, 0] = 255
                label_img_mat[i, j, 1] = 69
                label_img_mat[i, j, 2] = 0     # liver Red
                
            if pred_batch[i, j] == 0:
                pred_img_mat[i, j, 0] = pred_img_mat[i, j, 1] =  pred_img_mat[i, j, 2] = 128  # backgroud Gray
            if pred_batch[i, j] == 1:
                pred_img_mat[i, j, 0] = 255
                pred_img_mat[i, j, 1] = 69
                pred_img_mat[i, j, 2] = 0    # liver Red
    label_img_mat=np.uint8(label_img_mat)            
    pred_img_mat=np.uint8(pred_img_mat) 
#    scipy.misc.imsave(recordPath + 'imgs/' + '%d-mask.jpg' % (test_num), label_img_mat)
#    scipy.misc.imsave(recordPath + 'imgs/' + '%d-pred.jpg' % (test_num), pred_img_mat)
    
    scipy.misc.imsave(savepath + '%d-mask.bmp' % (test_num), label_img_mat)
    scipy.misc.imsave(savepath + '%d-pred.bmp' % (test_num), pred_img_mat)


#Accuracy_Measure
def rgb2Bin(img):
    res = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range (IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if img[i, j ,0] == 128 and img[i, j ,1] == 128 and img[i, j ,2] == 128:
                res[i,j] = 0
            if img[i, j ,0] == 255 and img[i, j ,1] == 69 and img[i, j ,2] == 0:
                res[i,j] = 1
    return res
    

def Accuracy_Measure(itr):
    print("Accuracy_Measure..........")
    res_path = recordPath+'imgs/'+'results'+str(itr)+'/'
    file_names = os.listdir(res_path)
    file_names.sort()
    res_num = len(file_names)
    
    gt_label = 0
    pr_label = 0
    TP = 0
    TN = 0
    
    for img_i in range(0, res_num, 2):
        label_batch = np.array(Image.open(res_path + file_names[img_i]))
        pred_batch = np.array(Image.open(res_path + file_names[img_i+1]))
        
        label = rgb2Bin(label_batch)
        pred = rgb2Bin(pred_batch)
              
        gt_label = gt_label + np.count_nonzero(label == 1)
        pr_label = pr_label + np.count_nonzero(pred == 1)
        
        label_bool = (label == 1)
        pred_bool = (pred == 1)
        common = np.logical_and(label_bool, pred_bool)
        TP = TP + np.count_nonzero(common == True)
        
        label_gro = (label == 0)
        pred_gro = (pred == 0)
        common2 = np.logical_and(label_gro, pred_gro)
        TN = TN + np.count_nonzero(common2 == True)
        #allpix = IMAGE_SIZE*IMAGE_SIZE*res_num//2
        
        
    dice_coe = 2*TP/(gt_label + pr_label)
    #MIOU = TP/(gt_label + pr_label-TP)
    #Pixel_Acc = (TP + TN)/allpix
    
  
    #print("DSC:", dice_coe)
   # print("MIOU:", MIOU)
    #print("PA:", Pixel_Acc)
   # print("GroundTruth_label", gt_label)
    #print("Predict", pr_label)
   # print("labPred", TP)
    return dice_coe

