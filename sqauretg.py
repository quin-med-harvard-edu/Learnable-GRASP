
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.compat.v1 as tff
import scipy.io as sio
import numpy as np
import os
import tensorflow.contrib.slim as slim
from time import time
from PIL import Image
import math
import pandas as pd
from numpy import array
from skimage.metrics import structural_similarity as ssimm

PhaseNumber_start=7
PhaseNumber_end=16
ch=32
data_dir='dataset_500'
EpochNum =100
batch_size = 64
ddelta = 0.01
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)  
mask = sio.loadmat('dataset_500/mask/mask_201.mat')
m = np.expand_dims(mask['mask_201'].astype(np.float32), axis=0)
m=tf.slice(m,[0,0,0],[batch_size,160,170])
m=tf.slice(m,[0,0,10],[batch_size,160,160])
kspace1 = tf.placeholder(tf.complex64, [None, 160, 160])#kspace1_rec
kspace2 = tf.placeholder(tf.complex64, [None, 160, 160])#kspace2_rec
rec_target1 = tf.placeholder(tf.float32, [None, 160, 160])
rec_target2 = tf.placeholder(tf.float32, [None, 160, 160])

def ssim(input, target, ksize=11, sigma=1.5, L=1.0):
    def ssimKernel(ksize=ksize, sigma=sigma):
        if sigma == None:  # no gauss weighting
            kernel = np.ones((ksize, ksize, 1, 1)).astype(np.float32)
        else:
            x, y = np.mgrid[-ksize // 2 + 1:ksize // 2 + 1, -ksize // 2 + 1:ksize // 2 + 1]
            kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
            kernel = kernel[:, :, np.newaxis, np.newaxis].astype(np.float32)
        return kernel / np.sum(kernel)

    kernel = tf.Variable(ssimKernel(), name='ssim_kernel', trainable=False)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu1_sqr = mu1 ** 2
    mu2_sqr = mu2 ** 2
    mu1mu2 = mu1 * mu2
    sigma1_sqr = tf.nn.conv2d(input * input, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu1_sqr
    sigma2_sqr = tf.nn.conv2d(target * target, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu2_sqr
    sigma12 = tf.nn.conv2d(input * target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC') - mu1mu2
    ssim_maps = ((2.0 * mu1mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sqr + mu2_sqr + C1) *
                                                                (sigma1_sqr + sigma2_sqr + C2))
    return tf.reduce_mean(tf.reduce_mean(ssim_maps, axis=(1, 2, 3)))
    
def add_con2d_weight(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    return Weights

def add_con2d_weight_k(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_k_%d' % order_no)
    return Weights

def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = tf.fft2d(img)
    # apply sampling mask
    kspace = tf.complex(tf.real(Fu) * sampling_mask, tf.imag(Fu) * sampling_mask)
    return kspace
    
def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    Finv = tf.ifft2d(tf.complex(tf.real(f) * sampling_mask, tf.imag(f) * sampling_mask))
    return Finv 
    
class Multi_modal_generator:

    def __init__(self, n):
        super(Multi_modal_generator,self).__init__()  
            
        with tf.variable_scope('lower', reuse=tf.AUTO_REUSE):
            self.alpha1_r1 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha1_i1 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha1   = tf.complex(self.alpha1_r1, self.alpha1_i1)
                    
            self.tau1_r1 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.tau1_i1 = tf.Variable(0.01, dtype=tf.float32, trainable=True)       
            self.tau1   = tf.complex(self.tau1_r1, self.tau1_i1)
                    
            self.shrinkage_thresh1 = tf.Variable(1E-8, dtype=tf.float32, trainable=True)
            
            self.alpha1_r2 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha1_i2 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha2   = tf.complex(self.alpha1_r2,self.alpha1_i2)
                    
            self.tau1_r2 = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.tau1_i2 = tf.Variable(0.01, dtype=tf.float32, trainable=True)       
            self.tau2   = tf.complex(self.tau1_r2, self.tau1_i2)
                    
            self.shrinkage_thresh2 = tf.Variable(1E-8, dtype=tf.float32, trainable=True)
            
            self.w1_1 = add_con2d_weight([3, 3, 2, ch], 11)
            self.w1_1_ = add_con2d_weight([3, 3, 2, ch], 110)
            self.w1_2 = add_con2d_weight([3, 3, ch, ch], 21)
            self.w1_2_ = add_con2d_weight([3, 3, ch, ch], 210)
            self.w1_3 = add_con2d_weight([3, 3, ch, ch], 31)
            self.w1_3_ = add_con2d_weight([3, 3, ch, ch], 310)
            self.w1_4 = add_con2d_weight([3, 3, ch, ch], 41)
            self.w1_4_ = add_con2d_weight([3, 3, ch, ch], 410)
        
    def sigma_activation(self,x_i):
        x_i_delta_sign = tf.sign(tf.nn.relu(tf.abs(x_i) - ddelta))
        x_square = tf.square(x_i)
        x_i_111 = tf.divide(1.0, 4.0 * ddelta)*x_square + 0.5*x_i + 0.25 * ddelta
        x_i_1 = tf.multiply(1.0 - x_i_delta_sign, x_i_111) + tf.multiply(x_i_delta_sign, tf.nn.relu(x_i))
        return x_i_1
    
    def sigma_derivative(self,x_i):
        x_i_delta_sign = tf.sign(tf.nn.relu(tf.abs(x_i) - ddelta))
        x_i_111 = tf.divide(1.0, 2.0 * ddelta)*x_i + 0.5
        x_i_1 = tf.multiply(1.0 - x_i_delta_sign, x_i_111) + tf.multiply(x_i_delta_sign, tf.sign(tf.nn.relu(x_i)))
        return x_i_1 
        
    def ssim(self,input, target, ksize=11, sigma=1.5, L=1.0):
        def ssimKernel(ksize=ksize, sigma=sigma):
            if sigma == None:  # no gauss weighting
                kernel = np.ones((ksize, ksize, 1, 1)).astype(np.float32)
            else:
                x, y = np.mgrid[-ksize // 2 + 1:ksize // 2 + 1, -ksize // 2 + 1:ksize // 2 + 1]
                kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
                kernel = kernel[:, :, np.newaxis, np.newaxis].astype(np.float32)
            return kernel / np.sum(kernel)
    
        kernel = tf.Variable(ssimKernel(), name='ssim_kernel', trainable=False)
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
    
        mu1 = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
        mu2 = tf.nn.conv2d(target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
        mu1_sqr = mu1 ** 2
        mu2_sqr = mu2 ** 2
        mu1mu2 = mu1 * mu2
        sigma1_sqr = tf.nn.conv2d(input * input, kernel, strides=[1, 1, 1, 1], padding='VALID',
                                  data_format='NHWC') - mu1_sqr
        sigma2_sqr = tf.nn.conv2d(target * target, kernel, strides=[1, 1, 1, 1], padding='VALID',
                                  data_format='NHWC') - mu2_sqr
        sigma12 = tf.nn.conv2d(input * target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC') - mu1mu2
        ssim_maps = ((2.0 * mu1mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sqr + mu2_sqr + C1) *
                                                                    (sigma1_sqr + sigma2_sqr + C2))
        return tf.reduce_mean(tf.reduce_mean(ssim_maps, axis=(1, 2, 3)))
    
    def w1(self, x1, x2):    
    
        x1 = tf.reshape(x1,shape=[batch_size,160,160,1])
        x2 = tf.reshape(x2,shape=[batch_size,160,160,1])
        x = tf.concat([x1, x2], axis=3) 
        x_imag = tf.imag(x)
        x_real = tf.real(x)
        # x1_real = tf.real(x1)
        # x1_imag = tf.imag(x1)
        # x2_real = tf.real(x2)
        # x2_imag = tf.imag(x2)
       
        # x_real = tf.concat([x1_real, x2_real], axis=-1)
        # x_imag = tf.concat([x1_imag, x2_imag], axis=-1) 
        
        w1_real = tf.nn.conv2d(x_real, self.w1_1, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x_imag, self.w1_1_, strides=[1, 1, 1, 1], padding='SAME')
        w1_imag = tf.nn.conv2d(x_real, self.w1_1_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x_imag, self.w1_1, strides=[1, 1, 1, 1], padding='SAME')
    
        w2_real = tf.nn.conv2d(self.sigma_activation(w1_real), self.w1_2, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w1_imag), self.w1_2_, strides=[1, 1, 1, 1], padding='SAME')
        w2_imag = tf.nn.conv2d(self.sigma_activation(w1_real), self.w1_2_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w1_imag), self.w1_2, strides=[1, 1, 1, 1], padding='SAME')
        
        w3_real = tf.nn.conv2d(self.sigma_activation(w2_real), self.w1_3, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w2_imag), self.w1_3_, strides=[1, 1, 1, 1], padding='SAME')
        w3_imag = tf.nn.conv2d(self.sigma_activation(w2_real), self.w1_3_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w2_imag), self.w1_3, strides=[1, 1, 1, 1], padding='SAME')
        
        w_real = tf.nn.conv2d(self.sigma_activation(w3_real), self.w1_4, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w3_imag), self.w1_4_, strides=[1, 1, 1, 1], padding='SAME') 
        w_imag = tf.nn.conv2d(self.sigma_activation(w3_real), self.w1_4_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w3_imag), self.w1_4, strides=[1, 1, 1, 1], padding='SAME') 
        #input image same dimension
        
        return w_real, w_imag
    
    def w2(self, x1, x2):    
    
        x1 = tf.reshape(x1,shape=[batch_size,160,160,1])
        x2 = tf.reshape(x2,shape=[batch_size,160,160,1])
        x = tf.concat([x1, x2], axis=3) 
        x_imag = tf.imag(x)
        x_real = tf.real(x)
        # x1_real = tf.real(x1)
        # x1_imag = tf.imag(x1)
        # x2_real = tf.real(x2)
        # x2_imag = tf.imag(x2)
       
        # x_real = tf.concat([x1_real, x2_real], axis=-1)
        # x_imag = tf.concat([x1_imag, x2_imag], axis=-1) 
        
        w1_real = tf.nn.conv2d(x_real, self.w1_1, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x_imag, self.w1_1_, strides=[1, 1, 1, 1], padding='SAME')
        w1_imag = tf.nn.conv2d(x_real, self.w1_1_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x_imag, self.w1_1, strides=[1, 1, 1, 1], padding='SAME')
    
        w2_real = tf.nn.conv2d(self.sigma_activation(w1_real), self.w1_2, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w1_imag), self.w1_2_, strides=[1, 1, 1, 1], padding='SAME')
        w2_imag = tf.nn.conv2d(self.sigma_activation(w1_real), self.w1_2_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w1_imag), self.w1_2, strides=[1, 1, 1, 1], padding='SAME')
        
        w3_real = tf.nn.conv2d(self.sigma_activation(w2_real), self.w1_3, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w2_imag), self.w1_3_, strides=[1, 1, 1, 1], padding='SAME')
        w3_imag = tf.nn.conv2d(self.sigma_activation(w2_real), self.w1_3_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w2_imag), self.w1_3, strides=[1, 1, 1, 1], padding='SAME')
        
        w_real = tf.nn.conv2d(self.sigma_activation(w3_real), self.w1_4, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(self.sigma_activation(w3_imag), self.w1_4_, strides=[1, 1, 1, 1], padding='SAME') 
        w_imag = tf.nn.conv2d(self.sigma_activation(w3_real), self.w1_4_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(self.sigma_activation(w3_imag), self.w1_4, strides=[1, 1, 1, 1], padding='SAME') 
        #input image same dimension
        t1 = tf.complex(w1_real,w1_imag)
        t2 = tf.complex(w2_real,w2_imag)
        t3 = tf.complex(w3_real,w3_imag)
        t4 = tf.complex(w_real,w_imag)
        return [t1,t2,t3,t4]
        
    def l2_norm_square(self,x):
        x = tf.square(tf.real(x)) + tf.square(tf.imag(x))
        x = tf.reduce_sum(x , axis =[-2,-1]) #shape=(batch_size,), dtype=float32)
        return x
        
    def w1_l21_delta(self, x1,x2, shrinkage_thresh):
        w1_real, w1_imag = self.w1(x1, x2)
        x = tf.complex(w1_real,w1_imag)
        # x = tf.norm(x, ord=2, axis=-1)
      
        x = tf.square(tf.real(x)) + tf.square(tf.imag(x))
        x = tf.reduce_sum(x , axis =-1) 
        x= tf.sqrt(x)
        
        s = shrinkage_thresh
        greater = tf.sign(tf.nn.relu(x-s))
        less = tf.ones_like(greater) - greater
        x = tf.multiply(less, tf.divide(tf.square(x), 2*s)) + tf.multiply(greater, x-s/2)
        x= tf.reshape(x, shape=[-1, 25600])
        y= tf.reduce_sum(x, 1, keep_dims=True)
        return y
        
    def w1gra(self,x1,x2, shrinkage_thresh):
        # x2=tf.zeros([1,160,180])
        output=self.w2( x1, x2)
        [t1,t2,t3,t4]=output
        
        t3_r = tf.reshape(tf.real(t3), shape=[batch_size, 160, 160, ch])
        t3_i = tf.reshape(tf.imag(t3), shape=[batch_size, 160, 160, ch])
        t2_r = tf.reshape(tf.real(t2), shape=[batch_size, 160, 160, ch])
        t2_i = tf.reshape(tf.imag(t2), shape=[batch_size, 160, 160, ch])
        t1_r = tf.reshape(tf.real(t1), shape=[batch_size, 160, 160, ch])
        t1_i = tf.reshape(tf.imag(t1), shape=[batch_size, 160, 160, ch])
        # x = tf.norm(x, ord=2, axis=-1)
      
        xnorm = tf.square(tf.real(t4)) + tf.square(tf.imag(t4))
        xnorm = tf.reduce_sum(xnorm , axis =-1) 
        xnorm= tf.sqrt(xnorm)

        s = shrinkage_thresh
        greater = tf.sign(tf.nn.relu(xnorm-s))
        greater = tf.tile(tf.expand_dims(tf.sign(tf.nn.relu(xnorm - s)), -1), [1, 1, 1, ch])
        less = tf.ones_like(greater) - greater
        
        tt=tf.complex(greater*tf.real(t4),greater*tf.imag(t4))
        ttnorm = tf.square(tf.real(tt)) + tf.square(tf.imag(tt))
        ttnorm = tf.reduce_sum(ttnorm , axis =-1) 
        ttnorm= tf.expand_dims(tf.sqrt(ttnorm),-1)
        ttnorm=tf.tile(ttnorm,[1,1,1,ch])
        x_i_1_greater=tf.complex(tf.divide(tf.real(tt),ttnorm),tf.divide(tf.imag(tt),ttnorm))
        
        # x_i_1_greater = tf.nn.l2_normalize(t4, dim=-1)
        ts=tf.complex(less*tf.real(t4), less*tf.imag(t4))
        x_i_1_less = tf.complex(tf.divide(tf.real(ts), shrinkage_thresh),tf.divide(tf.imag(ts), shrinkage_thresh))
        x_i_1_out = x_i_1_greater + x_i_1_less
        
        xxr = tf.reshape(tf.real(x_i_1_out), shape=[batch_size, 160, 160, ch])
        xxi = tf.reshape(tf.imag(x_i_1_out), shape=[batch_size, 160, 160, ch])
        
        x3_r_dec      = tf.nn.conv2d_transpose(xxr, self.w1_4, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose(xxi, self.w1_4_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        x3_i_dec      = tf.nn.conv2d_transpose(xxr, self.w1_4_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose(xxi, self.w1_4, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        
        x2_r_deri_act = self.sigma_derivative(t3_r)
        x2_i_deri_act = self.sigma_derivative(t3_i)
        
        x2_r_dec      = tf.nn.conv2d_transpose((x2_r_deri_act * x3_r_dec), self.w1_3, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose((x2_i_deri_act * x3_i_dec), self.w1_3_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        x2_i_dec      = tf.nn.conv2d_transpose((x2_r_deri_act * x3_r_dec), self.w1_3_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose((x2_i_deri_act * x3_i_dec), self.w1_3, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        
        x1_r_deri_act = self.sigma_derivative(t2_r)
        x1_i_deri_act = self.sigma_derivative(t2_i)

        x1_r_dec      = tf.nn.conv2d_transpose((x1_r_deri_act * x2_r_dec), self.w1_2, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose((x1_i_deri_act * x2_i_dec), self.w1_2_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        x1_i_dec      = tf.nn.conv2d_transpose((x1_r_deri_act * x2_r_dec), self.w1_2_, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose((x1_i_deri_act * x2_i_dec), self.w1_2, [batch_size, 160, 160, ch], [1, 1, 1, 1], padding='SAME')
        
        x0_r_deri_act = self.sigma_derivative(t1_r)
        x0_i_deri_act = self.sigma_derivative(t1_i)
        
        x0_r_dec      = tf.nn.conv2d_transpose((x0_r_deri_act * x1_r_dec), self.w1_1, [batch_size, 160, 160, 2], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose((x0_i_deri_act * x1_i_dec), self.w1_1_, [batch_size, 160, 160, 2], [1, 1, 1, 1], padding='SAME')
        x0_i_dec      = tf.nn.conv2d_transpose((x0_r_deri_act * x1_r_dec), self.w1_1_, [batch_size, 160, 160, 2], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose((x0_i_deri_act * x1_i_dec), self.w1_1, [batch_size, 160, 160, 2], [1, 1, 1, 1], padding='SAME')
        
        x0_r = tf.reshape(x0_r_dec,  shape=[batch_size, 160, 160, 2])
        x0_i = tf.reshape(x0_i_dec,  shape=[batch_size, 160, 160, 2]) 
        
        first_der_r = x0_r[:,:,:,0] 
        print(first_der_r.shape)
        first_der_i = x0_i[:,:,:,0]
        first_dec  = tf.complex(first_der_r, first_der_i) 
        
        second_der_r =  x0_r[:,:,:,1]
        print(second_der_r.shape)
        second_der_i = x0_i[:,:,:,1]
        second_dec  = tf.complex(second_der_r, second_der_i) 
    
        return first_dec, second_dec
        
    def data_fidelity1(x):
        pfx = mriForwardOp(x, m)    
        s = pfx - kspace1
        return 0.5*l2_norm_square(s)
    
    def data_fidelity2(x):
        pfx = mriForwardOp(x, m)    
        s = pfx - kspace2
        return 0.5*l2_norm_square(s)
        
    def phi(x1, x2, shrinkage_thresh):
        phi = data_fidelity1(x1) + data_fidelity2(x2) + w1_l21_delta(x1,x2,shrinkage_thresh)#shape=(2,), dtype=float32
        return phi
     
    def phigra_x1(x1, x2, shrinkage_thresh):
        y=phi(x1, x2, shrinkage_thresh)
        phix1=tf.gradients(y,x1)
       
        return phix1
        
    def phigra_x2(x1, x2, shrinkage_thresh):
        y=phi(x1, x2, shrinkage_thresh)
        phix2=tf.gradients(y,x2)
        return phix2  
        # gamma=0.9
        # sigma=1.0
    def block(self, input1, input2, alpha1, alpha2, tau1, tau2, shrinkage_thresh1):
        gamma=0.9
        sigma=1.0
        a1= 1E+3
        c1= 1E+3
        a2= 1E+3
        c2= 1E+3
        # SSS=xishu*shrinkage_thresh1
        input1 = tf.reshape(input1, shape=[batch_size, 160, 160]) 
        input2 = tf.reshape(input2, shape=[batch_size, 160, 160]) 
        
        first_dec1,second_dec1=self.w1gra( input1, input2,shrinkage_thresh1)
        
        ATAx1 = mriAdjointOp(mriForwardOp(input1, m), m)# FTPT(PFx1)
        ATf1  = mriAdjointOp(kspace1, m)#FTPT(f1)
        
        z1 = input1 - alpha1 * (ATAx1 - ATf1)
        u1 = z1 - tau1* first_dec1
        first_dec2,second_dec2= self.w1gra( u1, input2,shrinkage_thresh1)
        
        ATAx2 = mriAdjointOp(mriForwardOp(input2, m), m)# FTPT(PFx1)
        ATf2  = mriAdjointOp(kspace2, m)#FTPT(f1)
        
        z2 = input2 - alpha2 * (ATAx2 - ATf2)
        u2 = z2 - tau2*second_dec2

        phi_u = 0.5*self.l2_norm_square(mriForwardOp(u1, m)- kspace1) + 0.5*self.l2_norm_square(mriForwardOp(u2, m)- kspace1)+self.w1_l21_delta(u1,u2, shrinkage_thresh1)
        phi_x = 0.5*self.l2_norm_square(mriForwardOp(input1, m)- kspace1) + 0.5*self.l2_norm_square(mriForwardOp(input2, m)- kspace1)+self.w1_l21_delta(input1,input2, shrinkage_thresh1) #(batch_size, )
        phi_u1_x1 = phi_u - phi_x
        
        u1_x  = self.l2_norm_square(u1 - input1)+ self.l2_norm_square(u2 - input2)
        u1_x1 = tf.sqrt(self.l2_norm_square(u1 - input1))
        u2_x2 = tf.sqrt(self.l2_norm_square(u2 - input2))
        
        first_dec1t,second_dec1t=self.w1gra( u1, u2,shrinkage_thresh1)
        norm_grad_dec1t= tf.reduce_mean(tf.sqrt(self.l2_norm_square(first_dec1t)))
        norm_grad_dec1s= tf.reduce_mean(tf.sqrt(self.l2_norm_square(second_dec1t)))
        
        
        ooo=tf.logical_and((norm_grad_dec1t<= a1 * u1_x1), (norm_grad_dec1s<= a1 * u2_x2) )
        if ooo==True:
            oooo=1
        else:
            oooo=0
        
        if_choose_u1 = tf.logical_and((oooo==1), (phi_u1_x1 <= -1/a1 * u1_x))
        
        if if_choose_u1==True:
            x1 = u1
            x2 = u2
        else:
            def condition(alpha1):
                v1 = z1- alpha1*first_dec1
                first_dec3,second_dec3=self.w1gra( v1, input2,shrinkage_thresh1)
                v2 = z2 - alpha1*second_dec3
                
                phi_uu = 0.5*self.l2_norm_square(mriForwardOp(v1, m)- kspace1) + 0.5*self.l2_norm_square(mriForwardOp(v2, m)- kspace1)+self.w1_l21_delta(v1,v2, shrinkage_thresh1)
                phi_xx = 0.5*self.l2_norm_square(mriForwardOp(input1, m)- kspace1) + 0.5*self.l2_norm_square(mriForwardOp(input2, m)- kspace1)+self.w1_l21_delta(input1,input2, shrinkage_thresh1) #(batch_size, )
                phi_u1_x11 = phi_uu - phi_xx
                v1_x  = self.l2_norm_square(v1 - input1)+ self.l2_norm_square(v2 - input2)

                return [v1, v2, phi_u1_x11, v1_x]
                
            [v1, v2, phi_u1_x11, v1_x] = condition(alpha1)
    
            if_alpha1_decrease = tf.reshape(tf.reduce_mean(phi_u1_x11) <= tf.reduce_mean( c1 * v1_x), [])
            #check if alpha should decrease
            if if_alpha1_decrease==False:
                alpha1  = tf.where(if_alpha1_decrease, alpha1, tf.complex(0.9*tf.real(alpha1),0.9*tf.imag(alpha1)) )
                alpha2  = tf.where(if_alpha1_decrease, alpha2, tf.complex(0.9*tf.real(alpha2),0.9*tf.imag(alpha2)) )
                [v1, v2, phi_v1_x11, v1_x] = condition(alpha1)
            x1 = v1
            x2=v2
            
        # alpha1  = tf.where(if_alpha1_decrease, alpha1, tf.complex(0.9*tf.real(alpha1),0.9*tf.imag(alpha1)) )
        return [x1, x2, alpha1, alpha2, tau1, tau2, shrinkage_thresh1]
        
    def forward(self, n, reuse):
        # global alpha1
        global alpha2
        global tau1
        global tau2
        global shrinkage_thresh1
        global shrinkage_thresh2
        layers1 = []        
        layers2=[]
        # x1_initial = mriAdjointOp(kspace1, m) #(?, 160, 180)
        layers1.append(kspace1)
        layers2.append(kspace2)
        a1 = []
        t1 = []
        e1 = []        
        a1.append(self.alpha1)
        t1.append(self.tau1)
        e1.append(self.shrinkage_thresh1)
        
        a2 = []
        t2 = []
            
        a2.append(self.alpha2)
        t2.append(self.tau2)
      
        for i in range(n):     
            [x1, x2, alpha1, alpha2, tau1, tau2, shrinkage_thresh1]= self.block(layers1[-1], layers2[-1], a1[-1], a2[-1], t1[-1], t2[-1],e1[-1])
            
            layers1.append(x1)
            a1.append(self.alpha1)
            t1.append(tau1)
            e1.append(shrinkage_thresh1)     
            layers2.append(x2)
            a2.append(alpha2)
            t2.append(tau2)
                
        return layers1[-1],e1,layers2[-1]
        
    # def forward(self, n, reuse):      
    #     rec1_layers = []
    #     rec2_layers = []
        
    #     rec1_layers.append(x1_0)
    #     rec2_layers.append(x2_0)
        
    #     for i in range(n):
    #         with tf.variable_scope('conv_%d' %i, reuse=reuse):
    #             rec1, rec2 = self.block(rec1_layers[-1], rec2_layers[-1], i)
    #             rec1_layers.append(rec1)
    #             rec2_layers.append(rec2)
    #     return tf.abs(rec1_layers[-1]), tf.abs(rec2_layers[-1])
        
def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs if t is not None])

def MSE(y_true, y_pred):
    return tf.reduce_mean((tf.abs( y_true) - tf.abs(y_pred)) ** 2)

def MAE(y_true, y_pred):
    return tf.reduce_mean(tf.abs( y_true - y_pred))

def compute_cost(img_rec1, img_rec2, rec_target1, rec_target2):
    
    img_rec11=tf.abs(img_rec1)
    rec_target11=tf.abs(rec_target1)

    img_rec11 = tf.expand_dims(img_rec11, -1)
    rec_target11=tf.expand_dims(rec_target11, -1)
    # img_rec11 = tf.squeeze(img_rec11, 0)
    # rec_target11=tf.squeeze(rec_target11, 0)
    

    # img1 = np.reshape(img_rec11, (160,180)).astype(np.float32)
    # imgg1 = np.reshape(rec_target11, (160,180)).astype(np.float32)
    img_rec22=tf.abs(img_rec2)
    rec_target22=tf.abs(rec_target2)
    img_rec22 = tf.expand_dims(img_rec22, -1)
    rec_target22=tf.expand_dims(rec_target22, -1)

    ssim1= ssim(img_rec11, rec_target11)
    # ssim1=ssimm(img1,imgg1)
    ssim3= ssim(img_rec22, rec_target22)

    # cost_tr_rec1 = MSE(rec_target1[0:1], img_rec1[0:1]) + (1-ssim1) 
    # cost_tr_rec2 = MSE(rec_target2[0:1], img_rec2[0:1]) + (1-ssim3)

    # cost_vl_rec1 = MSE(rec_target1[1:2], img_rec1[1:2]) + (1-ssim2)
    # cost_vl_rec2 = MSE(rec_target2[1:2], img_rec2[1:2]) + (1-ssim4)
    cost_tr_rec1 = MSE(rec_target1, img_rec1) + (1-ssim1) 
    cost_tr_rec2 = MSE(rec_target2, img_rec2) + (1-ssim3)

   
    train_loss = cost_tr_rec1 + cost_tr_rec2
    # val_loss = cost_vl_rec1 + cost_vl_rec2
    
    return train_loss, cost_tr_rec1, cost_tr_rec2

# model = Multi_modal_generator(PhaseNumber)
# img_rec1, e1, img_rec2= model.forward(PhaseNumber, reuse=tf.AUTO_REUSE)
# train_loss, cost_tr_rec1, cost_tr_rec2 = compute_cost(img_rec1, img_rec2,rec_target1,rec_target2) 

# learning_rate = tf.train.exponential_decay(learning_rate = 0.0001,
#                                       global_step = global_step,
#                                       decay_steps = 100,
#                                       decay_rate=0.9, staircase=False) 

# theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lower')

# stop_ct = l2norm_sq((tf.gradients(train_loss, theta)))

# Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, name='Adam_%d' % PhaseNumber 
# trainer_1 = Optimizer.minimize(train_loss, global_step=global_step, var_list=theta)#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
# sess = tf.Session(config=config)
# init = tf.global_variables_initializer()
# sess.run(init)
 

model_dir = 'weight_flair_6' 
log_file_name = "Log_%s.txt" % (model_dir)
#___________________________________________________________________________________Train

print("...................................")
# print("Phase Number is %d" % (PhaseNumber))
print("...................................\n")
print('Load Data...')
        
x1_0='T1_0train.mat'
x2_0='T2_0train.mat'
        
X1_0 = sio.loadmat(x1_0)
X2_0 = sio.loadmat(x2_0)
X1_0.pop('__header__')
X1_0.pop('__version__')
X1_0.pop('__globals__')
X2_0.pop('__header__')
X2_0.pop('__version__')
X2_0.pop('__globals__')
        # X1_0 = sio.loadmat(x1_0)['x1_0'].astype(np.complex64)
        # X2_0 = sio.loadmat(x2_0)['x2_0'].astype(np.complex64)
        
        
df1= tf.convert_to_tensor(X1_0['T1_0train'])
df2= tf.convert_to_tensor(X2_0['T2_0train'])

df1=tf.slice(df1,[0,0,0],[2600,160,170])
df1=tf.slice(df1,[0,0,10],[2600,160,160])
df2=tf.slice(df2,[0,0,0],[2600,160,170])
df2=tf.slice(df2,[0,0,10],[2600,160,160])

df2 = df2.eval(session=tf.compat.v1.Session())
df1 = df1.eval(session=tf.compat.v1.Session())

        # K1 = './%s/train/K1.mat'%(data_dir)
        # K1_ = './%s/train/K1_.mat'%(data_dir)
        
        # K2 = './%s/train/K2.mat'%(data_dir)
        # K2_ = './%s/train/K2_.mat'%(data_dir)
        
T1 = './%s/train/T1.mat'%(data_dir)
T1_ = './%s/train/T1_.mat'%(data_dir)
        
T2 = './%s/train/T2.mat'%(data_dir)
T2_ = './%s/train/T2_.mat'%(data_dir)
        
        # k1 = sio.loadmat(K1)['K1'].astype(np.complex64)
        # k1_ = sio.loadmat(K1_)['K1_'].astype(np.complex64)
        # k1_train = np.vstack((k1, k1_))#(1300, 160, 180)
        
        # k2 = sio.loadmat(K2)['K2'].astype(np.complex64)
        # k2_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
        # k2_train = np.vstack((k2, k2_))#(1300, 160, 180)
        
gt_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
gt_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
gt_train_T1 = np.vstack((gt_T1, gt_T1_))
        
gt_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
gt_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
gt_train_T2 = np.vstack((gt_T2, gt_T2_))
        
ntrain = df1.shape[0]
print('Total training data %d' % ntrain)
        
        # K1 = './%s/validation/K1.mat'%(data_dir)
        # K1_ = './%s/validation/K1_.mat'%(data_dir)
        
        # K2 = './%s/validation/K2.mat'%(data_dir)
        # K2_ = './%s/validation/K2_.mat'%(data_dir)
        
T1 = './%s/validation/T1.mat'%(data_dir)
T1_ = './%s/validation/T1_.mat'%(data_dir)
        
T2 = './%s/validation/T2.mat'%(data_dir)
T2_ = './%s/validation/T2_.mat'%(data_dir)
        
        
        # FLAIR = './%s/validation/FLAIR.mat'%(data_dir)
        # FLAIR_ = './%s/validation/FLAIR_.mat'%(data_dir)
        
        # k1_v = sio.loadmat(K1)['K1'].astype(np.complex64)
        # k1_v_ = sio.loadmat(K1_)['K1_'].astype(np.complex64)
        # k1_val = np.vstack((k1_v, k1_v_))
        
        # k2_v = sio.loadmat(K2)['K2'].astype(np.complex64)
        # k2_v_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
        # k2_val = np.vstack((k2_v, k2_v_))
        
gt_v_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
gt_v_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
gt_val_T1 = np.vstack((gt_v_T1, gt_v_T1_))
        
gt_v_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
gt_v_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
gt_val_T2 = np.vstack((gt_v_T2, gt_v_T2_))
        
        # gt_v_FLAIR = sio.loadmat(FLAIR)['FLAIR'].astype(np.float32)
        # gt_v_FLAIR_ = sio.loadmat(FLAIR_)['FLAIR_'].astype(np.float32)
        # gt_val_FLAIR = np.vstack((gt_v_FLAIR, gt_v_FLAIR_))
        
gt_train_T11 = np.vstack((gt_T1, gt_T1_,gt_v_T1, gt_v_T1_))
gt_train_T22 = np.vstack((gt_T1, gt_T1_,gt_v_T1, gt_v_T1_))
gt_train_T11=tf.slice(gt_train_T11,[0,0,0],[2600,160,170])
gt_train_T11=tf.slice(gt_train_T11,[0,0,10],[2600,160,160])
gt_train_T22=tf.slice(gt_train_T22,[0,0,0],[2600,160,170])
gt_train_T22=tf.slice(gt_train_T22,[0,0,10],[2600,160,160])
        
for PhaseNumber in range(PhaseNumber_start, PhaseNumber_end, 2):
    if PhaseNumber > 12:
        batch_size = 32
    model = Multi_modal_generator(PhaseNumber)
    img_rec1, e1, img_rec2= model.forward(PhaseNumber, reuse=tf.AUTO_REUSE)
    train_loss, cost_tr_rec1, cost_tr_rec2 = compute_cost(img_rec1, img_rec2,rec_target1,rec_target2) 

    learning_rate = tf.train.exponential_decay(learning_rate = 0.0001,
                                       global_step = global_step,
                                       decay_steps = 100,
                                       decay_rate=0.9, staircase=False) 

    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lower')

    stop_ct = l2norm_sq((tf.gradients(train_loss, theta)))

    Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, name='Adam_%d' % PhaseNumber 
    trainer_1 = Optimizer.minimize(train_loss, global_step=global_step, var_list=theta)#
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
            
    if PhaseNumber > 3:
        EpochNum = 100
    else:
        EpochNum = 300
    for epoch_i in range(EpochNum+1):
        randidx_all = np.random.permutation(ntrain)
        for batch_i in range(ntrain//batch_size):
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
           
            print(gt_train_T11.shape)
            t1_true = gt_train_T11[randidx,:,:]
            
            t2_true = gt_train_T22[randidx,:,:]
      
            xio=df1[randidx, :, :]
            xip=df2[randidx, :, :]
            print("koop")
            print(xio.shape)
            print("kppp")
            
            feed_dict = {kspace1: xio, kspace2: xip, rec_target1: t1_true, rec_target2: t2_true} 
                # feed_dict = {kspace1: k1, kspace2: kflair, rec_target1: t1_true, rec_target2: flair_true} #kspace不要
                
                # delta = sess.run(stop_ct, feed_dict=feed_dict)
                # if (delta > 0.001):
            self.session.run(trainer_1, feed_dict=feed_dict)
                # sess.run(trainer_2, feed_dict=feed_dict)
                
            output_data = "[%02d%02d/%02d/%02d] cost_tr_rec1: %.7f, cost_tr_rec2: %.7f \n" % (PhaseNumber,epoch_i, EpochNum, batch_i, self.session.run(cost_tr_rec1, feed_dict=feed_dict), self.session.run(cost_tr_rec2, feed_dict=feed_dict)) # rec1: %.7f, rec2: %.7f, sess.run(w_tr2, feed_dict=feed_dict)
            print(output_data)
                
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if batch_i % 10 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d_%d.ckpt' % (PhaseNumber,model_dir, epoch_i, batch_i), write_meta_graph=False)
                
        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if epoch_i % 10 == 0:
            saver.save(self.session, './%s/CS_Saved_Model_%d_1360.ckpt' % (PhaseNumber, model_dir, epoch_i), write_meta_graph=False)
        

print("Training Finished")   

#____________________________________________________________________________________Test
print('Load Testing Data...')

K1_space = './%s/test/K1.mat'%(data_dir)
K2_space = './%s/test/K2.mat'%(data_dir)
T1 = './%s/test/T1.mat'%(data_dir)
T2 = './%s/test/T2.mat'%(data_dir)
FLAIR = './%s/test/FLAIR.mat'%(data_dir)

k1_test = sio.loadmat(K1_space)['K1'].astype(np.complex64)
k2_test = sio.loadmat(K2_space)['K2'].astype(np.complex64)
gt_test_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
gt_test_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
gt_test_FLAIR = sio.loadmat(FLAIR)['FLAIR'].astype(np.float32)
ntest = k1_test.shape[0]

TIME_ALL  = []
PSNR1_All = []
SSIM1_All = []
NMSE1_All = []
PSNR2_All = []
SSIM2_All = []
NMSE2_All = []
PSNR3_All = []
SSIM3_All = []
NMSE3_All = []

saver.restore(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, ckpt_model_number))

result_file_name = "Meta_Rec_Results.txt"
    
idx_all = np.arange(ntest)
for imag_no in range(ntest):
    randidx = idx_all[imag_no:imag_no+1]
    kspace_t1 = k1_test[randidx, :, :]
    kspace_t2 = k2_test[randidx, :, :]
    target_t1 = gt_test_T1[randidx, :, :]
    target_t2 = gt_test_T2[randidx, :, :]
    target_flair = gt_test_FLAIR[randidx, :, :]

    feed_dict = {kspace1: kspace_t1, kspace2: kspace_t2, rec_target1: target_t1, rec_target2: target_t2}
    
    start = time()
    Reconstruction1_value = sess.run(img_rec1, feed_dict=feed_dict) 
    Reconstruction2_value = sess.run(img_rec2, feed_dict=feed_dict) 
    end = time()
    
    rec_t1 = np.reshape(Reconstruction1_value, (160,180)).astype(np.float32)
    rec_t2 = np.reshape(Reconstruction2_value, (160,180)).astype(np.float32)
    #syn_flair = np.reshape(Synthesis_value, (160,180)).astype(np.float32)
    rec_reference_t1 = np.reshape(target_t1, (160,180)).astype(np.float32)
    rec_reference_t2 = np.reshape(target_t2, (160,180)).astype(np.float32)
    #syn_reference_flair = np.reshape(target_flair, (160,180)).astype(np.float32)
    
    PSNRt1 =  psnr(rec_reference_t1, rec_t1)
    SSIMt1 =  ssim(rec_reference_t1, rec_t1)
    NMSEt1 =  nmse(rec_reference_t1, rec_t1)

    PSNRt2 =  psnr(rec_reference_t2, rec_t2)
    SSIMt2 =  ssim(rec_reference_t2, rec_t2)
    NMSEt2 =  nmse(rec_reference_t2, rec_t2)

    #PSNRt3 =  psnr(syn_reference_flair, syn_flair)
    #SSIMt3 =  ssim(syn_reference_flair, syn_flair)
    #NMSEt3 =  nmse(syn_reference_flair, syn_flair)
    
    result1 = "Run time for %s:%.4f, PSNRt1:%.4f, SSIMt1:%.4f, NMSEt1:%.4f. \n" % (imag_no+1, (end - start), PSNRt1, SSIMt1, NMSEt1)
    result2 = "Run time for %s:%.4f, PSNRt2:%.4f, SSIMt2:%.4f, NMSEt2:%.4f. \n" % (imag_no+1, (end - start), PSNRt2, SSIMt2, NMSEt2)
    #result3 = "Run time for %s:%.4f, PSNRt3:%.4f, SSIMt3:%.4f, NMSEt3:%.4f. \n" % (imag_no+1, (end - start), PSNRt3, SSIMt3, NMSEt3)
    
    print(result1)
    print(result2) 
    im_rec_name = "%s_rec_%d.mat" % (imag_no+1, ckpt_model_number)  
    # save mat file
    #Utils.saveAsMat(rec, im_rec_name, 'result',  mat_dict=None)
    
    PSNR1_All.append(PSNRt1)
    SSIM1_All.append(SSIMt1)
    NMSE1_All.append(NMSEt1)

    PSNR2_All.append(PSNRt2)
    SSIM2_All.append(SSIMt2)
    NMSE2_All.append(NMSEt2)
    
    TIME_ALL.append((end - start))
 
output_data1 = "rec_t1 ckpt NO. is %d, Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, time: %.4f\n" % (ckpt_model_number, np.mean(PSNR1_All), np.std(PSNR1_All), np.mean(SSIM1_All), np.std(SSIM1_All), np.mean(NMSE1_All), np.std(NMSE1_All), np.mean(TIME_ALL))
print(output_data1)
output_data2 = "rec_t2 ckpt NO. is %d, Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, time: %.4f\n" % (ckpt_model_number, np.mean(PSNR2_All), np.std(PSNR2_All), np.mean(SSIM2_All), np.std(SSIM2_All), np.mean(NMSE2_All), np.std(NMSE2_All), np.mean(TIME_ALL))
print(output_data2)
#output_data3 = "syn_flair ckpt NO. is %d, Avg PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, time: %.4f\n" % (ckpt_model_number, np.mean(PSNR3_All), np.std(PSNR3_All), np.mean(SSIM3_All), np.std(SSIM3_All), np.mean(NMSE3_All), np.std(NMSE3_All), np.mean(TIME_ALL))
#print(output_data3)

output_file = open(result_file_name, 'a')
output_file.write(output_data1)
output_file.write(output_data2)
#output_file.write(output_data3)
output_file.close()
sess.close()