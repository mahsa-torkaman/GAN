# author is He Zhao
# The time to create is 8:47 PM, 28/11/16

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Opts import lrelu, resUnit
from dataBlocks import DataBlocks
import cPickle
import scipy.io as sio
import numpy as np
import pdb


initializer = tf.truncated_normal_initializer(stddev=0.02)
bias_initializer = tf.constant_initializer(0.0)


def build_data(batchsize):
    readpath = open('img2.pkl', 'rb')
    #pdb.set_trace()
    datapaths = cPickle.load(readpath)

    #------------------------------------------
    #If different .pkl file was used with number of blocks=20
    i=0
    new_list=[]#break the datapath list to a list of lists
    while i<len(datapaths):
      new_list.append(datapaths[i:i+2])
      i+=2
    #-------------------------------------------
    db = DataBlocks(data_paths=datapaths, train_valid_ratio=[39, 0], batchsize=batchsize, allow_preload=False)
    
    #mask_s = np.load('syn_data/mask_trn20+tst15+flip.npy')
    mask_s = np.load('syn_data/mask_trn20_scaled.npy')
    return db, mask_s
   


def matTonpy_35():
    #M_images,M_masks,M_manual1
    img = sio.loadmat('test_1To4_double2.mat')['M_images']
    gt = sio.loadmat('test_1To4_double2.mat')['M_manual1']
    mask = sio.loadmat('test_1To4_double2.mat')['M_masks']

    return img, gt, mask

def discriminator(image, reuse=False):
    n=32
    bn = slim.batch_norm
    with tf.name_scope("disciminator"):
        # original
        dis1 = slim.convolution2d(image, n, [4, 4], 2, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)

        dis2 = slim.convolution2d(dis1, 2*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)

        dis3 = slim.convolution2d(dis2, 4*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)

        dis4 = slim.convolution2d(dis3, 8*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv4', weights_initializer=initializer)

        dis5 = slim.convolution2d(dis4, 16*n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv5', weights_initializer=initializer)

        
        d_out_logits = slim.fully_connected(slim.flatten(dis5), 1, activation_fn=None, reuse=reuse, scope='d_out',
                                            weights_initializer=initializer)

        d_out = tf.nn.sigmoid(d_out_logits)
    return d_out, d_out_logits
    
def generator(image, z):
    n = 64
    with tf.name_scope("generator"):
        # original
        e1 = slim.conv2d(image, n, [4, 4], 2, activation_fn=lrelu, scope='g_e1_conv',
                         weights_initializer=initializer)
        # 256
        e2 = slim.conv2d(lrelu(e1), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e2_conv',
                         weights_initializer=initializer)
        # 128
        e3 = slim.conv2d(lrelu(e2), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e3_conv',
                         weights_initializer=initializer)
        # 64
        e4 = slim.conv2d(lrelu(e3), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e4_conv',
                         weights_initializer=initializer)
        # 32
        e5 = slim.conv2d(lrelu(e4), 8*n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e5_conv',
                         weights_initializer=initializer)
        # 16
        e6 = slim.conv2d(lrelu(e5), 8*n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None, scope='g_e6_conv',
                         weights_initializer=initializer)



        zP = slim.fully_connected(z, 4 * 4 * n, normalizer_fn=None, activation_fn=lrelu, scope='g_project',
                                  weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 4, 4, n])

        gen1 = slim.conv2d_transpose(lrelu(zCon), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv1', weights_initializer=initializer)
        # 8
        gen1 = tf.concat([gen1, e6], 3)
        
        gen2 = slim.conv2d_transpose(lrelu(gen1), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv2', weights_initializer=initializer)
        # 16
        gen2 = tf.concat([gen2, e5], 3)

        gen3 = slim.conv2d_transpose(lrelu(gen2), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv3', weights_initializer=initializer)
        gen3 = tf.concat([gen3, e4], 3)

        # 32
        gen6 = slim.conv2d_transpose(tf.nn.relu(gen3), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv6', weights_initializer=initializer)
        gen6 = tf.concat([gen6, e3], 3)

        # 64
        gen7 = slim.conv2d_transpose(tf.nn.relu(gen6), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv7', weights_initializer=initializer)
        gen7 = tf.concat([gen7, e2], 3)

        # 128
        gen8 = slim.conv2d_transpose(tf.nn.relu(gen7), n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv8', weights_initializer=initializer)
        # gen8 = tf.nn.dropout(gen8, 0.5)
        gen8 = tf.concat([gen8, e1], 3)
        gen8 = tf.nn.relu(gen8)

        # 256
        gen_out = slim.conv2d_transpose(gen8, 3, [4, 4], 2, activation_fn=tf.nn.tanh, scope='g_out',
                                        weights_initializer=initializer)

    return gen_out    
    
