#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 22:00:22 2018

@author: Niny
"""

'''
read from .raws
'''
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.lib.io import file_io as file_io


cwd = os.getcwd()
print(cwd)

#recordPath = '/media/xenia/Niny_Q/ISELS2018/Unet/'
recordPath = os.getcwd()+'/Data/'

shape=[256,256]

def read_tfrcd(filename):
    file_queue = tf.train.string_input_producer([filename])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
        
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'img_raw': tf.FixedLenFeature([], tf.string),
                    'gt_raw': tf.FixedLenFeature([], tf.string),
                    'example_name': tf.FixedLenFeature([], tf.string)
                    })
    
    with tf.variable_scope('decoder'):
        depth = features['depth']
        image = tf.decode_raw(features['img_raw'], tf.float32)
        ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)
        example_name = features['example_name']
        
        
    with tf.variable_scope('image'):
        # reshape and add 0 dimension (would be batch dimension)
        image = tf.cast(tf.reshape(image, shape), tf.float32)
        image = tf.expand_dims(image, axis = 2)
    with tf.variable_scope('ground_truth'):
        # reshape
        ground_truth = tf.cast(tf.reshape(ground_truth, shape), tf.float32)
        ground_truth = tf.expand_dims(ground_truth, axis = 2)
    return image,ground_truth,example_name,depth



def read_and_decord(mode):
    if mode=='Train':
        filename = recordPath+'DWITrainArgu10.tfrecord'
        image,ground_truth,example_name,depth=read_tfrcd(filename)
    if mode=='Test':
        filename = recordPath+'DWITest.tfrecord'
        image,ground_truth,example_name,depth=read_tfrcd(filename)
        
   
    return image,ground_truth,example_name,depth


    

if __name__ == '__main__':
	#createTrainRecord()

	# img_data shape: [batch_size, depths, row(height), cols(width)] ,
	# need to add channels =1 at the end of cols(width) to 5D tensor
    
    
    image,label,name,depth= read_and_decord('Train')
    
    img_batch,label_batch = tf.train.batch([image,label], batch_size = 1, capacity=1)
    init = tf.global_variables_initializer()
    local=tf.local_variables_initializer()
    
    sess=tf.Session() 
    sess.run([init,local])
    sum=0
    threads = tf.train.start_queue_runners(sess)
    for i in range(500):
        sum=sum+1
        x,y= sess.run([img_batch, label_batch])
        n,s= sess.run([name,depth])
        print (n,s)
        #print(x.shape)
        #print(y.shape)
        #print (s)
        #print(sum)
		#np.savetxt('xx32.csv', x[0, 39, :, :], delimiter=',', fmt = '%.4f' )










