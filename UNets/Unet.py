#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 04:58:35 2018

@author: Niny
"""

import tensorflow as tf
from .Layers import convolution_layer, deconvolution_layer, batch_norm

FLAGS = tf.app.flags.FLAGS

NUM_OF_CLASS = 2


"""
2Dunet 
image_size=256*256
input_batch=[batch_size,256,256,1]

Batch_norm
strid=2 convolution replaced max_pooling
"""

def u_net(in_tensor, is_training=True):
    n_channels=64
    
    with tf.variable_scope('encoder_level_1'):
        with tf.variable_scope('conv_1'):
            c11 = convolution_layer(in_tensor, [3,3, 1, n_channels], [1,1,1,1])
            c11_bn = batch_norm(c11,is_training)
            c11_relu = tf.nn.relu(c11_bn)
            
        with tf.variable_scope('conv_2'):
            c12 = tf.nn.relu(convolution_layer(c11_relu, [3,3,n_channels, n_channels],[1,1,1,1]))
            c12_bn = batch_norm(c12,is_training)
            c12_relu = tf.nn.relu(c12_bn)
            
        with tf.variable_scope('down_sample'):
            c13 = tf.nn.relu(convolution_layer(c12_relu, [2,2, n_channels, n_channels*2], [1,2,2,1]))
            
   
    with tf.variable_scope('encoder_level_2'):
        with tf.variable_scope('conv_1'):
            c21 = convolution_layer(c13, [3,3,n_channels*2, n_channels*2],[1,1,1,1])
            c21_bn = batch_norm(c21,is_training)
            c21_relu = tf.nn.relu(c21_bn)

        with tf.variable_scope('conv_2'):
            c22 = tf.nn.relu(convolution_layer(c21_relu, [3,3,n_channels*2, n_channels*2],[1,1,1,1]))
            c22_bn = batch_norm(c22,is_training)
            c22_relu = tf.nn.relu(c22_bn)
            
        with tf.variable_scope('down_sample'):
            c23 = tf.nn.relu(convolution_layer(c22_relu, [2,2, n_channels*2, n_channels*4], [1,2,2,1]))
            
    
    with tf.variable_scope('encoder_level_3'):
        with tf.variable_scope('conv_1'):
            c31 = convolution_layer(c23, [3,3,n_channels*4, n_channels*4],[1,1,1,1])
            c31_bn = batch_norm(c31,is_training)
            c31_relu = tf.nn.relu(c31_bn)
            
        with tf.variable_scope('conv_2'):
            c32 = convolution_layer(c31_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1])
            c32_bn = batch_norm(c32,is_training)
            c32_relu = tf.nn.relu(c32_bn)
            
        with tf.variable_scope('down_sample'):
            c33 = tf.nn.relu(convolution_layer(c32_relu, [2,2, n_channels*4, n_channels*8], [1,2,2,1]))
     
        
    with tf.variable_scope('encoder_level_4'):
        with tf.variable_scope('conv_1'):
            c41 = convolution_layer(c33, [3,3,n_channels*8, n_channels*8],[1,1,1,1])
            c41_bn = batch_norm(c41,is_training)
            c41_relu = tf.nn.relu(c41_bn)
            
        with tf.variable_scope('conv_2'):
            c42 = convolution_layer(c41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1])
            c42_bn = batch_norm(c42,is_training)
            c42_relu = tf.nn.relu(c42_bn)
            
        with tf.variable_scope('down_sample'):
            c43 = tf.nn.relu(convolution_layer(c42_relu, [2,2, n_channels*8, n_channels*16], [1,2,2,1]))
    
    
    with tf.variable_scope('encoder_decoder_level_5'):
        with tf.variable_scope('conv_1'):
            c51 = convolution_layer(c43, [3,3,n_channels*16, n_channels*16],[1,1,1,1])
            c51_bn = batch_norm(c51,is_training)
            c51_relu = tf.nn.relu(c51_bn)
            
        with tf.variable_scope('conv_2'):
            c52 = convolution_layer(c51_relu, [3,3,n_channels*16, n_channels*16],[1,1,1,1])
            c52_bn = batch_norm(c52,is_training)
            c52_relu = tf.nn.relu(c52_bn)
            
        with tf.variable_scope('up_sample'):
            c53 = deconvolution_layer(c52_relu, [2,2,n_channels*8, n_channels*16], tf.shape(c42), [1,2,2,1])
            c53_relu = tf.nn.relu(c53)
            
            
    with tf.variable_scope('decoder_level_4'):
        d4 = tf.concat((c53_relu, c42), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d41 = convolution_layer(d4, [3,3,n_channels*16, n_channels*8],[1,1,1,1])
            d41_bn = batch_norm(d41,is_training)
            d41_relu = tf.nn.relu(d41_bn)
            
        with tf.variable_scope('conv_2'):
            d42 = convolution_layer(d41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1])
            d42_bn = batch_norm(d42,is_training)
            d42_relu = tf.nn.relu(d42_bn)
            
        with tf.variable_scope('up_sample'):
            d43 = deconvolution_layer(d42_relu, [2,2,n_channels*4, n_channels*8], tf.shape(c32), [1,2,2,1])
            d43_relu = tf.nn.relu(d43)
            
    with tf.variable_scope('decoder_level_3'):
        d3 = tf.concat((d43_relu, c32), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d31 = convolution_layer(d3, [3,3,n_channels*8, n_channels*4],[1,1,1,1])
            d31_bn = batch_norm(d31,is_training)
            d31_relu = tf.nn.relu(d31_bn)
        
        with tf.variable_scope('conv_2'):
            d32 = convolution_layer(d31_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1])
            d32_bn = batch_norm(d32,is_training)
            d32_relu = tf.nn.relu(d32_bn)
            
        with tf.variable_scope('up_sample'):
            d33 = deconvolution_layer(d32_relu, [2,2,n_channels*2, n_channels*4], tf.shape(c22), [1,2,2,1])
            d33_relu = tf.nn.relu(d33)
        
            
    with tf.variable_scope('decoder_level_2'):
        d2 = tf.concat((d33_relu, c22), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d21 = convolution_layer(d2, [3,3,n_channels*4, n_channels*2],[1,1,1,1])
            d21_bn = batch_norm(d21,is_training)
            d21_relu = tf.nn.relu(d21_bn)
        
        with tf.variable_scope('conv_2'):
            d22 = convolution_layer(d21_relu, [3,3,n_channels*2, n_channels*2],[1,1,1,1])
            d22_bn = batch_norm(d22,is_training)
            d22_relu = tf.nn.relu(d22_bn)
        
        with tf.variable_scope('up_sample'):
            d23 = deconvolution_layer(d22_relu, [2,2,n_channels, n_channels*2], tf.shape(c12), [1,2,2,1])
            d23_relu = tf.nn.relu(d23)
            
            
    with tf.variable_scope('decoder_level_1'):
        d1 = tf.concat((d23_relu, c12), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d11 = convolution_layer(d1, [3,3,n_channels*2, n_channels],[1,1,1,1])
            d11_bn = batch_norm(d11,is_training)
            d11_relu = tf.nn.relu(d11_bn)
            
        with tf.variable_scope('conv_2'):
            d12 = convolution_layer(d11_relu, [3,3,n_channels, n_channels],[1,1,1,1])
            d12_bn = batch_norm(d12,is_training)
            d12_relu = tf.nn.relu(d12_bn)
            
    with tf.variable_scope('output_layer'):
        logits = convolution_layer(d12_relu, [1,1,n_channels,NUM_OF_CLASS], [1,1,1,1])
        
    
    annotation_pred = tf.expand_dims(tf.argmax(logits, dimension = 3, name = 'prediction'), dim= 3)
    
    return logits, annotation_pred
            
            
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.Debug:
        # print(len(var_list))
        for grad, var in grads:
            tf.summary.histogram(grad, var)
            
    return optimizer.apply_gradients(grads)
            
            
            
            
            
            
            