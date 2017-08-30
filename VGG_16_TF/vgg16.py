########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
# Edited by: Sajjad Mozaffari 15-08-2017                                               #
########################################################################################
'''
Edits:
* Python Version changed to 3.6
* TensorFlow variable definition changed.
* Trainable param in tf.variable set to off
* Visualizations are improved
'''

import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from imagenet_classes import class_names
import matplotlib.pyplot as plt

def weight_variable(name, shape,dtype = tf.float32, stddev = 0.1, trainable = False):
    return tf.get_variable(name, shape, dtype, tf.truncated_normal_initializer(stddev = stddev), trainable = trainable)

def bias_variable(name, shape, dtype = tf.float32, constant = 0, trainable = False):
    return tf.get_variable(name, shape, dtype, tf.constant_initializer(constant), trainable = trainable)


class vgg16:
    def __init__(self, imgs, initial_weight=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if initial_weight is not None and sess is not None:
            self.load_weights(initial_weight, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.variable_scope('preprocess'):
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.variable_scope('conv1_1'):
            kernel = weight_variable('weights', [3, 3, 3, 64])
            biases = bias_variable('biases', [64])
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2'):
            kernel = weight_variable('weights', [3, 3, 64, 64])
            biases = bias_variable('biases', [64])
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1'):
            kernel = weight_variable('weights', [3, 3, 64, 128])
            biases = bias_variable('biases', [128])
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2'):
            kernel = weight_variable('weights', [3, 3, 128, 128])
            biases = bias_variable('biases', [128])
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1'):
            kernel = weight_variable('weights', [3, 3, 128, 256])
            biases = bias_variable('biases', [256])
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2'):
            kernel = weight_variable('weights', [3, 3, 256, 256])
            biases = bias_variable('biases', [256])
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3'):
            kernel = weight_variable('weights', [3, 3, 256, 256])
            biases = bias_variable('biases', [256])
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1'):
            kernel = weight_variable('weights', [3, 3, 256, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2'):
            kernel = weight_variable('weights', [3, 3, 512, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3'):
            kernel = weight_variable('weights', [3, 3, 512, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1'):
            kernel = weight_variable('weights', [3, 3, 512, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2'):
            kernel = weight_variable('weights', [3, 3, 512, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3'):
            kernel = weight_variable('weights', [3, 3, 512, 512])
            biases = bias_variable('biases', [512])
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.variable_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = weight_variable('weights', [shape, 4096])
            fc1b = bias_variable('biases', [4096], 1.0)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.variable_scope('fc2'):
            fc2w = weight_variable('weights', [4096, 4096], trainable=True)
            fc2b = bias_variable('biases', [4096], 1.0, trainable=True)

            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.variable_scope('fc3'):
            fc3w = weight_variable('weights', [4096, 1000], trainable=True)
            fc3b = bias_variable('biases', [1000], 1.0, trainable=True)

            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        initial_weight = np.load(weight_file)
        keys = sorted(initial_weight.keys())
        print(keys)
        for i, k in enumerate(keys):
            print(i, k, np.shape(initial_weight[k]))
            sess.run(self.parameters[i].assign(initial_weight[k]))

if __name__ == '__main__':
    with tf.Session() as sess:
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
        img1 = imread('example.jpg')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:4]
        string = ""
        for p in preds:
            string += class_names[p]+str(prob[p])+'\n'
        plt.figure(figsize=(6, 6))
        plt.imshow(img1)
        plt.text(0,260,string)
        plt.axis('off')
        plt.show()