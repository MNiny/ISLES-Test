#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:15:33 2018


@author: Niny
"""

from DataProcessor import Readtf as read_mat

import tensorflow as tf 
import numpy as np
from scipy import misc

import TFUtils as utils
#from UNets import Unet_Res as net
from PSPUNet import Psp_Unet as net

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.app.flags.DEFINE_integer("batch_size", '2', 'batch size of training')
tf.app.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.app.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.app.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.app.flags.DEFINE_bool('Debug', "False", "Debug mode: True/ False")
tf.app.flags.DEFINE_string('Mode', "Train", "Mode Train/ Test/ Visualize")

FLAGS = tf.app.flags.FLAGS

recordPath = os.getcwd()+'/'

NUM_OF_CLASS = 2
IMAGE_SIZE=256

model_save = 500
test_epoch = 2500

Epoch = 400
batch_sz=8
Train_NUM = 2000

MAX_ITERATION = 100000

IMAGE_CHANNEL = 1 

TEST_RAW =96




def main(agrv = None):
	#FLAGS.para_name
    
    #keep_probability = tf.placeholder(tf.float32, name = "keep_prob") #dropout: keep_probability
    is_training = tf.placeholder(tf.bool, name = 'is_train') #BN: istraining
    
       
    image = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, 1], name = 'input_img')    
    annotation = tf.placeholder(tf.int32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, 1], name = 'annotation')
    
    
    #logits, pred = net.u_net(image, is_training)
    
    logits, pred = net.PSPUnet(image, is_training)
    
    tf.summary.image("image", image, max_outputs = batch_sz)
    tf.summary.image("groud_truth", tf.cast(annotation, tf.uint8), max_outputs = batch_sz)
    tf.summary.image("pred_annotation", tf.cast(pred, tf.uint8), max_outputs = batch_sz)
    
    #softmax cross entropy loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = tf.squeeze(annotation, axis=[3]),name = 'entropy_loss')))
    
    
    
    
    #mix loss(dice+cross entropy)
    #loss = utils.mixed_loss(logits, annotation)
    
    #sigmoid focal loss
    #loss = utils.focal_loss(logits, annotation)
    
    #one-hot focal loss
    #anno_squ=tf.squeeze(annotation,axis=4)
    #anno_onehot=tf.one_hot(anno_squ,2)
    #loss = utils.focal_loss(logits,anno_onehot)
    
    #dice=utils.dice_similarity_coefficient(tf.cast(pred,tf.int32), annotation)
    
    
    
    loss_train_op = tf.summary.scalar('training_loss', loss)
    
    Testloss = tf.placeholder(tf.float32, name = 'Test_loss')
    loss_test_op = tf.summary.scalar('test_loss',Testloss)
    
    DSC_batch = tf.placeholder(tf.float32, name = 'Dice_coeff')
    DSC_op = tf.summary.scalar('Dice_coefficient', DSC_batch)
    
    trainable_var = tf.trainable_variables()
    if FLAGS.Debug:
        for var in trainable_var:
            tf.summary.histogram(var.op.name, var)
            tf.add_to_collection('reg_loss', tf.nn.l2_loss(var))
    
    #BN: update moving_mean&moving_variance
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = net.train(loss, trainable_var)
    
    print("setting up image reader...")
    image_batch_a, label_batch_a, name, depth = read_mat.read_and_decord('Train')
    
    img_train_batch, label_train_batch = tf.train.shuffle_batch([image_batch_a, label_batch_a],
																	batch_size=batch_sz, capacity=batch_sz*2, min_after_dequeue=batch_sz)

    print (img_train_batch.shape)
    print ('setup session...')
    print("Setting up dataset reader")
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=50)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir,sess.graph)
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess)
    test_i = -1
    
    if FLAGS.Mode == 'Train':
        for itr in range (MAX_ITERATION):
            #img_batch shape:[batch_size, depth, height, width, chanl], type: ndarray

            img_batch, l_batch = sess.run([img_train_batch, label_train_batch ])
            
            print (itr, img_batch.shape)
            feed = {image: img_batch, annotation: l_batch, is_training: True}
            sess.run(train_op, feed_dict = feed)
            train_loss_print, summary_str = sess.run([loss, loss_train_op],  feed_dict = feed)
            print (train_loss_print)
            summary_writer.add_summary(summary_str, itr)
            
            if itr%model_save == 0 or itr%5000 == 0: 
                saver.save(sess, './mod/model', global_step=itr)
            elif itr == (MAX_ITERATION - 1):
                saver.save(sess, './mod/model', global_step=itr)
    
            ############################# test test test test test test test test test test#####################################
            #if itr == (MAX_ITERATION - 1):
            if (itr!=0 and itr%test_epoch == 0) or itr == (MAX_ITERATION - 1):
                #overfittest
                test_i = test_i+1
                print("train finish! start test~")
                
                #checkpoint_dir='Q:/Codes/ISLES2018/Unet/mod/'
                #checkpoint_dir = recordPath+'mod/'
                #modulefile=tf.train.latest_checkpoint(checkpoint_dir)
                #saver.restore(sess,os.path.join(checkpoint_dir,modulefile))
                
                #filename = 'Q:/mods/model-95000'
                #saver.restore(sess,filename)
                
                test_img_data_a, test_label_data_a, test_name, test_depth = read_mat.read_and_decord('Test')
                
                test_img_train_batch, test_label_train_batch = tf.train.batch([test_img_data_a, test_label_data_a],batch_size=1, capacity=1)
                
                threads = tf.train.start_queue_runners(sess)
                Testl = 0.0
                for test_itr in range (TEST_RAW): 
                    test_img_batch, test_l_batch = sess.run([test_img_train_batch, test_label_train_batch])
                
                    test_feed = {image: test_img_batch, annotation: test_l_batch, is_training: False}
                    
                    test_pred_logits, pred_image, testloss = sess.run([logits, pred, loss], feed_dict = test_feed)
                    
                    Testl = Testl+testloss
                    #pred_image = sess.run(tf.argmax(input=test_pred_logits,axis=3))
                    
                    
                    label_batch = np.squeeze(test_l_batch)
                    pred_batch = np.squeeze(pred_image)
                    #label_batch_tp = np.transpose(label_batch, (0, 2, 1))
                    label_tosave = np.reshape(label_batch, [IMAGE_SIZE,IMAGE_SIZE])
                    #pred_batch_tp = np.transpose(pred_batch, (0, 2, 1))
                    pred_tosave = np.reshape(pred_batch, [IMAGE_SIZE,IMAGE_SIZE])
                    print("test_itr:",test_itr)
                    # tep  = test_pred_annotation[0, 30, :, 0]
                    #np.savetxt('pred30.csv', tep, delimiter=',')
                    #np.savetxt('dice_smi_co.csv',test_dice_coe, delimiter=',')
                    utils.save_imgs(test_itr, label_tosave, pred_tosave, itr)
                Testl = Testl/TEST_RAW
                test_summary_str = sess.run(loss_test_op,  feed_dict = {Testloss:Testl})
                print (test_i,':',Testl)
                summary_writer.add_summary(test_summary_str, test_i)
                
                #Dise similarity coefficient
                DSC = utils.Accuracy_Measure(itr)
                DSC_Summary_str = sess.run(DSC_op,  feed_dict = {DSC_batch:DSC})
                print (test_i,':',DSC)
                summary_writer.add_summary(DSC_Summary_str, test_i)

                
    elif FLAGS.Mode == 'Visualize':
        pass
    print ('finished!')


if __name__ == '__main__':
	tf.app.run()
