# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:31:08 2018

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
    

def Accuracy_Measure():
    print("Accuracy_Measure..........")
    res_path = recordPath+'imgs/'
    file_names = os.listdir(res_path)
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
        allpix = IMAGE_SIZE*IMAGE_SIZE*res_num//2
        
        
    dice_coe = 2*TP/(gt_label + pr_label)
    MIOU = TP/(gt_label + pr_label-TP)
    Pixel_Acc = (TP + TN)/allpix
    
  
    print("DSC:", dice_coe)
    print("MIOU:", MIOU)
    print("PA:", Pixel_Acc)
    print("GroundTruth_label", gt_label)
    print("Predict", pr_label)
    print("labPred", TP)
    return dice_coe,MIOU,Pixel_Acc

