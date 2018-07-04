#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 22:38
# @Author  : SuperDan
# @Site    : 
# @File    : CIFAR10_inference.py
# @Software: PyCharm

import tensorflow as tf
import CIFAR10_input

# 配置神经网络的参数
# Global constants describing the CIFAR-10 data set
IMAGE_SIZE = CIFAR10_input.IMAGE_SIZE                                                         # 输入层的图片大小
NUM_CHANNELS = 3                                                                              # 输入层的深度
NUM_LABELS = CIFAR10_input.NUM_CLASSES                                                        # 分类一共10类

INPUT_NODE = 576
OUTPUT_NODE = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC1_SIZE = 384
FC2_SIZE = 192

REGULARAZTION_RATE = 0.004

# 定义卷积神经网络的前向传播过程。这里添加一个参数train，用于区分训练过程和测试过程。在这个程序中将用到dropout方法，
# 这个方法可以进一步提升模型可靠性并防止过拟合，这个方法只在训练过程使用。
def inference(input_tensor, train, regularizer='L2'):

    if regularizer == 'L2':
        regularize = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    else:
        regularize = tf.contrib.layers.l1_regularizer(REGULARAZTION_RATE)
    # 声明第一层卷积层的变量并实现前向传播过程。通过使用不同的命名空间来隔离不同层的变量，这可以让每一层的变量命名
    # 只需要考虑在当前层的作用，而不需要担心重名的问题。和标准LENET-5模型不太一样，这里卷积层输入为24*24*3的原始
    # CIFAR10图片像素。因为卷积层中使用了全0填充，所以输出为24*24*64的矩阵

    with tf.variable_scope('layer1-conv1'):
        with tf.device('/cpu:0'):
            conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=
                                        tf.truncated_normal_initializer(stddev=0.01))
            conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        tf.summary.histogram('conv1_weights', conv1_weights)

    # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，使用全0填充且移动的步长为2.这一层的输入
    # 是上一层的输出，24*24*64，输出为12*12*64
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    # LRN层——Local Response Normalization,即局部响应归一化层。神经网络学习过程本质就是为了学习数据分布，一旦训练数据
    # 与测试数据的分布不同，那么网络的泛化能力也大大降低；另一方面，每批训练数据的分布各不相同，那么网络就要在每次迭代都
    # 去学习适应不同的分布，这样将会大大降低网络的训练速度。
    with tf.name_scope('layer3_norm1'):
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    # 声明第四层卷积层的变量并实现前向传播过程。输入为12*12*64，输出为12*12*64
    with tf.variable_scope('layer4-conv2'):
        with tf.device('/cpu:0'):
            conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=
                                        tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且使用全0填充。
        conv2 = tf.nn.conv2d(norm1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        tf.summary.histogram('conv2_weights', conv2_weights)

    # norm2
    with tf.name_scope('layer5-norm2'):
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')

    # 实现第六层池化层的前向传播过程。输入为12*12*64， 输出为6*6*64
    with tf.name_scope('layer6-pool2'):
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第七层全连接层的变量并实现前向传播过程。这一层的输入时拉直之后的一组向量。这一层和之前介绍的全连接层
    # 基本一致，唯一区别是引入了dropout的概念。dropout在训练时会随机将部分节点的输出改为0。dropout可以避免过拟合问题
    # 从而使得模型在测试数据上的效果更好。dropout一般只在全连接层而不是卷积层或者池化层使用。
    with tf.variable_scope('layer7-fc1'):
        # 将第六层池化层输出转化为第七层全连接层的输入格式。第四层的输出为6*6*64的矩阵，第七层全连接层需要的输入格式
        # 为向量，所以需要将这个三维矩阵拉直成一个一维向量。pool2.get_shape函数可以得到第四层输出矩阵的维度。
        # 因为每一层神经网络的输入输出为一个batch的矩阵，所以这里得到的维度包含了一个batch中数据的个数
        pool_shape = pool2.get_shape().as_list()
        # 计算将矩阵拉直成向量之后的长度，长度就是矩阵长宽及深度的乘积。注意pool_shape[0]为一个batch中数据的个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

        # 通过tf.reshape函数将第四层的输出变成一个batch的向量
        reshaped = tf.reshape(pool2, [-1, nodes])
        with tf.device('/cpu:0'):
            fc1_weights = tf.get_variable('weight', [nodes, FC1_SIZE], initializer=
                                           tf.truncated_normal_initializer(stddev=0.04))
            fc1_biases = tf.get_variable('bias', [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('fc1_weights', fc1_weights)

        # 只有全连接层的权重需要加入正则化
        if train:
            tf.add_to_collection('losses', regularize(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # if train:
        #     fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第八层全连接层的变量并实现前向传播过程。输入为384长度的向量，输出为192的向量。
    with tf.variable_scope('layer8-fc2'):
        with tf.device('/cpu:0'):
            fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE], initializer=
                                           tf.truncated_normal_initializer(stddev=0.04))
            fc2_biases = tf.get_variable('bias', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('fc2_weights', fc2_weights)

        # 只有全连接层的权重需要加入正则化
        if train:
            tf.add_to_collection('losses', regularize(fc2_weights))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        # if train:
        #     fc2 = tf.nn.dropout(fc2, 0.5)

    # 声明第九层sotmax层的变量并实现前向传播过程。输入为192长度的向量，输出为10的向量。
    with tf.variable_scope('layer9-softmax'):
        with tf.device('/cpu:0'):
            softmax_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS], initializer=
                                          tf.truncated_normal_initializer(stddev=1/192.0))
            softmax_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('softmax_weights', softmax_weights)

        # if train:
        #     tf.add_to_collection('losses', regularize(softmax_weights))
        logit = tf.matmul(fc2, softmax_weights) + softmax_biases
    return logit
