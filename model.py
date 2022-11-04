from __future__ import print_function
import os
import time
import random
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import *

class LLE(object):
    def __init__(self, sess):
        self.sess = sess
        self.base_lr = 0.001
        self.input_low = tf.placeholder(tf.float32, [None, None, None,1], name='input_low') 
        self.input_high = tf.placeholder(tf.float32, [None, None, None,1], name='input_high')
        self.output = LiCENt(self.input_low)
        self.loss = tf.reduce_mean(1-tf.abs(tf.image.ssim(self.output , self.input_high,max_val = 1.0)))               
        self.global_step = tf.Variable(0, trainable = False)
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step, 100, 0.96)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")
        
    def train(self, train_low_data, train_high_data,  batch_size, patch_size, epoch, ckpt_dir):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        load_model_status, global_step = self.load(self.saver, ckpt_dir)
        if load_model_status:
            iter = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter))

        start_time = time.process_time()
        image_id = 0
        total_loss = 0
        epoch_loss = 0.000001

        for epoch in range(start_epoch, epoch):
            total_loss = 0.000001
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size,1), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size,1), dtype="float32")
                for patch_id in range(batch_size):
                
                    h, w , _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)           
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :,:] = data_aug(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :,:] = data_aug(train_high_data[image_id][x : x+patch_size, y : y+patch_size,:], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)
                # train

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input_low: batch_input_low, self.input_high: batch_input_high}) 

                total_loss = total_loss + loss
                epoch_loss = total_loss/numBatch
  
                print("Epoch: [%2d] [%4d/%4d]" % (epoch + 1, batch_id + 1, numBatch))    
                iter += 1
            print(" Epoch [%2d], time : %4.4f ,SSIM_loss : %.6f "%(epoch + 1,time.process_time() - start_time , epoch_loss))
            self.save(self.saver, iter, ckpt_dir, "LiCENt")
        print("[*] Finish training , total training time : time : %4.4f "%(time.process_time() - start_time))

    def save(self, saver, iter, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess,os.path.join(ckpt_dir, model_name),global_step=iter)
        
    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data_l, test_low_data,test_low_data_names, save_dir):
        tf.global_variables_initializer().run()
        print("[*] Reading checkpoint...")       
        load_model_status, _ = self.load(self.saver, './model/')
        if load_model_status:
            print("[*] Load weights successfully...")       
        print("[*] Testing...")
        total_time = 0.0
        start_time = time.process_time()
        for idx in range(len(test_low_data)):
            cost_time = 0.0
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            input_low_test = np.expand_dims(test_low_data_l[idx], axis=0) 
            start_time1 = time.process_time()
            result = self.sess.run(self.output, feed_dict = {self.input_low: input_low_test})
            result = np.squeeze(result)
            l = result*255
            hls = test_low_data[idx]            
            h = cv2.split(hls)[0]
            s = cv2.split(hls)[2]  #new
            hls_en = np.dstack([h,l,s])	
            hls_en = np.clip(hls_en, 0, 255)
            hls_en = hls_en.astype(np.uint8)
            post_img = cv2.cvtColor(hls_en,cv2.COLOR_HLS2BGR)			
            cost_time = time.process_time()-start_time1
            total_time = total_time + cost_time
            print("No.[%d/%d]" % (idx+1,len(test_low_data)))			
            save_img(os.path.join(save_dir, name + "." +suffix), post_img)          
        ave_run_time = total_time / float(len(test_low_data))
        print("[*] Average run time: %.4f" % ave_run_time)
        
# Network Architecture
def LiCENt(input_im):                        #low-light enhancer model
  with tf.variable_scope('LiCENt'):
    input = tf.image.resize_images(input_im, (192, 192),method = 0)     #192
    # BE Layer
    # Encoder part
    conv1 = tf.layers.conv2d(input,  16, 3, 2, padding='same', activation=tf.nn.relu) # 96
    conv2 = tf.layers.conv2d(conv1,  16, 3, 2, padding='same', activation=tf.nn.relu) # 48
    conv3 = tf.layers.conv2d(conv2,  16, 3, 2, padding='same', activation=tf.nn.relu) # 24
    conv4 = tf.layers.conv2d(conv3,  16, 3, 2, padding='same', activation=tf.nn.relu) # 12
    conv5 = tf.layers.conv2d(conv4,  16, 3, 2, padding='same', activation=tf.nn.relu) # 6
    conv6 = tf.layers.conv2d(conv5,  16, 3, 2, padding='same', activation=tf.nn.relu) # 3
    conv7 = tf.layers.conv2d(conv6,  16, 3, 2, padding='same', activation=tf.nn.relu) # 1
    # Decoder part
    deconv0 = tf.image.resize_images(conv7, (3, 3),method = 0)
    deconv0 = tf.layers.conv2d(deconv0, 16, 3, 1, padding='same', activation=tf.nn.relu)    
    deconv0 = tf.concat([deconv0, conv6], axis=3)  
    deconv1 = tf.image.resize_images(deconv0, (6, 6),method =0)
    deconv1 = tf.layers.conv2d(deconv1, 16, 3, 1, padding='same', activation=tf.nn.relu)
    deconv1 = tf.concat([deconv1, conv5], axis=3)  
    deconv2 = tf.image.resize_images(deconv1, (12, 12),method = 0)
    deconv2 = tf.layers.conv2d(deconv2, 16, 3, 1, padding='same', activation=tf.nn.relu)
    deconv2 = tf.concat([deconv2, conv4], axis=3)  
    deconv3 = tf.image.resize_images(deconv2, (24, 24),method = 0)
    deconv3 = tf.layers.conv2d(deconv3, 16, 3, 1, padding='same', activation=tf.nn.relu)
    deconv3 = tf.concat([deconv3, conv3], axis=3) 
    deconv4 = tf.image.resize_images(deconv3, (48, 48),method = 0)
    deconv4 = tf.layers.conv2d(deconv4, 16, 3, 1, padding='same', activation=tf.nn.relu)
    deconv4 = tf.concat([deconv4, conv2], axis=3) 
    deconv5 = tf.image.resize_images(deconv4, (96, 96),method = 0)
    deconv5 = tf.layers.conv2d(deconv5, 16, 3, 1, padding='same', activation=tf.nn.relu)
    deconv5 = tf.concat([deconv5, conv1], axis=3) 
    deconv6 = tf.image.resize_images(deconv5, (192, 192),method = 0) 
    deconv6 = tf.layers.conv2d(deconv6, 16, 3, 1, padding='same', activation=tf.nn.relu)   
    deconv6 = tf.concat([deconv6, input], axis=3)
    output = tf.image.resize_images(deconv6, (tf.shape(input_im)[1], tf.shape(input_im)[2]),method = 0)
    a_input = tf.concat([output, input_im], axis=3)
    #  DE Layer
    re_conv1 = tf.layers.conv2d(a_input,  64, 3, 1, padding='same', activation=tf.nn.relu)
    re_conv2 = tf.layers.conv2d(re_conv1, 64, 3, 1, padding='same', activation=tf.nn.relu)
    re_conv3 = tf.layers.conv2d(re_conv2, 64, 3, 1, padding='same', activation=tf.nn.relu)
    re_conv4 = tf.layers.conv2d(re_conv3, 64, 3, 1, padding='same', activation=tf.nn.relu) 
    # Mix channel
    output = tf.layers.conv2d(re_conv4, 1, 3, 1, padding='same', activation=tf.nn.relu)
    return output
