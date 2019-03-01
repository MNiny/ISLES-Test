#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:01:38 2018


@author: Niny
"""

import numpy as np
import nibabel as nib
import os
import tensorflow as tf

TRAIN, TEST = 'TRAIN', 'TEST'

train_inpath = 'Q:/DATA/Brain/2018Test/Train'
#train_inpath = 'Q:/DATA/DWI/ISLES2018/savetest'
test_inpath = 'Q:/DATA/Brain/2018Test/Test'
outpath = 'Q:/Codes/ISLES2018/ISLES/Data/'
#outpath = 'Q:/Codes/ISLES2018/test/'



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





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



def get_sub_folders(folder):
    return [sub_folder for sub_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub_folder))]

def get_image_type_from_folder_name(folder_name):
    image_types = ['.MR_DWI', '.OT']
    return next(image_type for image_type in image_types if image_type in folder_name)

def get_extension(filename):
    filename, extension = os.path.splitext(filename)
    return extension

def build_multimodal_image(image_list):
    shape = image_list[0].shape
    for image in image_list:
        assert image.shape == shape
    return np.stack(image_list).astype(np.float32)


def data_tfsave(input_dir, output_dir, mode=TRAIN):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # ADC are raw data, OT is the expert segmentation
    if mode == TRAIN:
        filename='DWITrain.tfrecord'
    elif mode == TEST:
        filename='DWITest.tfrecord'
    modes_to_use = ['.MR_DWI']
    data = {}
    label = {}
    
    print(mode)
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir+filename))
    for folder in get_sub_folders(input_dir):
        print(folder)
        data[folder] = {}
        label[folder] = {}
        sha=[]
        key = bytes(folder,encoding="utf8")  
        
        for sub_folder in get_sub_folders(os.path.join(input_dir, folder)):
            image_type = get_image_type_from_folder_name(sub_folder)
            
            path = os.path.join(input_dir, folder, sub_folder)
            filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
            path = os.path.join(path, filename)
            
            im = nib.load(path)
            image = im.get_data()
            shape = image.shape
            sha=shape
            
            for depth in range(shape[2]):
                img=image[:,:,depth]
                if image_type == '.OT':
                    label[folder][depth] = img.astype(np.uint8)
                if image_type in modes_to_use:
                   # _buffer.append(img)
                    norm_img=contrast_normalization(img)
                    data[folder][depth] = build_multimodal_image(norm_img)
        for depth in range(sha[2]):
            #print(depth)
            img_save = data[folder][depth]
            gt_save = label[folder][depth]
            #print(img_save.shape)
            #print(gt_save.shape)
            example = tf.train.Example(features=tf.train.Features(feature={
                        'example_name': _bytes_feature(key),
                        'depth': _int64_feature(depth),
                        'img_raw': _bytes_feature(img_save.tostring()),
                        'gt_raw': _bytes_feature(gt_save.tostring())}))
            writer.write(example.SerializeToString())
    writer.close()
       
        
if __name__ == '__main__':           
    data_tfsave(train_inpath, outpath)
   #data_tfsave(test_inpath, outpath, mode=TEST)
