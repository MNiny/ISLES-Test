# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 19:50:22 2018

@author: Niny
"""


import numpy as np
import nibabel as nib
import os
import tensorflow as tf
from PIL import Image
import cv2


def contrast_normalization(image, min_divisor=1e-3):
    """
    Data normalization
    
     output = (input-mean)/Standard_deviation
    
    """
    mean = image.mean()
    std = image.std()
    if std < min_divisor:
        std = min_divisor
    return (image - mean) / std

def data_argument(image, mode):

    if mode == 'False':
        res = image
        
    if mode=='Mirror':
        Im = Image.fromarray(image)
        res = Im.transpose(Image.FLIP_LEFT_RIGHT)
        res = np.array(res)
          
    if mode == 'Rot90':
        Im = Image.fromarray(image)
        res = Im.rotate(90)
        res = np.array(res)
    
    if mode == 'Rot-90':
        Im = Image.fromarray(image)
        res = Im.rotate(-90)
        res = np.array(res)
    return res

def Ran_displ(image, dx,dy):    
    tran = np.float32([[1,0,dx],[0,1,dy]])
    rows,cols = image.shape[:2]
    res = cv2.warpAffine(image,tran,(rows,cols))
    res = np.array(res)
    return res