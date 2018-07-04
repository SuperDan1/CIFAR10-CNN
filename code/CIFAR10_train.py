#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/21 21:18
# @Author  : SuperDan
# @Site    : 
# @File    : CIFAR10_train.py
# @Software: PyCharm
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
# 加载CIFAR10_inference.py中定义的常量和前向传播的函数
import numpy as np
import CIFAR10_inference
import CIFAR10_input
from datetime import datetime

import sys
import tarfile
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
# 配置神经网络的参数
"""
    tf.app.flags主要用于处理命令行参数的解析工作，其实可以理解为一个
    封装好的argparse包（一种结构化的数据存储格式，类似于json、xml）。
    使用方法：首先通过tf.app.flags来调用flags.py文件，之后用
    flags.DEFINE_interger/float来添加命令行参数，而FLAGS=tf.app.flags.FLAGS
    可以实例化这个解析参数的类从对应的命令行参数取出参数
"""
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = CIFAR10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('BATCH_SIZE', 128,
                            """训练批次大小""")
tf.app.flags.DEFINE_float('LEARNING_RATE_BASE', 0.1,
                            """初始的学习率""")
tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY', 0.9,
                            """学习率下降速率""")
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                            """滑动模型参数""")
tf.app.flags.DEFINE_integer('train_epoch', 100,
                            """训练轮数""")
tf.app.flags.DEFINE_integer('TRAINING_STEPS', int(FLAGS.train_epoch * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 128),
                            """总共数据大小""")
tf.app.flags.DEFINE_bool('log_device_placement', False,
                            """是否打印设备分配情况""")

# 数据地址
data_dir = '../data-bin'

# 模型保存的路径和文件名
MODEL_SAVE_PATH = '../model/'
MODEL_NAME = 'model.ckpt'

def CIFAR10_train():
    # 将处理输入数据的计算都放在名字为'input'的命名空间下
    with tf.name_scope('input'):
        # 读取数据
        images_train, lables_train = CIFAR10_input.distorted_inputs( data_dir=data_dir, batch_size=FLAGS.BATCH_SIZE)
        images_test, lables_test = CIFAR10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=FLAGS.BATCH_SIZE)
        # 定义输入输出placeholder
        x = tf.placeholder(tf.float32,[None, CIFAR10_inference.IMAGE_SIZE, CIFAR10_inference.IMAGE_SIZE,
                                       CIFAR10_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, CIFAR10_inference.OUTPUT_NODE], name='y-input')

    # 使用LeNet5_inference定义的前向传播
    y = CIFAR10_inference.inference(x, True, 'L2')
    global_step = tf.Variable(0, trainable=False)

    # 将处理滑动平均相关的计算都放在一个命名空间下
    with tf.name_scope('moving_average'):
        # 定义滑动平均操作
        variable_average = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY, global_step)
        variables_average_op = variable_average.apply(tf.trainable_variables())

    # 将计算损失函数相关的计算都放在一个命名空间下
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss_function', loss)

    # 将定义学习率、优化方法以及每一轮训练需要执行的操作放在一个命名空间
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE_BASE,global_step,50000 / FLAGS.BATCH_SIZE,
                                                   FLAGS.LEARNING_RATE_DECAY, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
        # 顺序执行
        with tf.control_dependencies([train_step, variables_average_op]):
            train_op = tf.no_op(name='train')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.summary.scalar('accuracy_train', accuracy_train)

        accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # tf.summary.scalar('accuracy_test', accuracy_test)

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess, ui_type="readline")  # 被调试器封装的会话
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 合并日志
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../log_train', tf.get_default_graph())
        xs_test, ys_test = sess.run([images_test, lables_test])
        # 对标签进行onehot编码
        ys_test_onehot = np.eye(10, dtype=float)[ys_test]
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程会有一个独立的程序来完成
        for i in range(FLAGS.TRAINING_STEPS):
            xs, ys = sess.run([images_train, lables_train])
            # 对标签进行onehot编码
            ys_onehot = np.eye(10, dtype=float)[ys]

            # 每1000轮保存一次模型
            if i % 1000 == 0 :
                # 配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
                _, loss_value, step, result = sess.run([train_op, loss, global_step, merged],
                                                       feed_dict={x: xs, y_: ys_onehot},
                                                       options=run_options, run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                writer.add_summary(result, i)
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解
                # 训练的情况。在验证集上的正确率信息会有一个单独的程序来 生成。

                train_accuracy = accuracy_train.eval(feed_dict={x: xs, y_: ys_onehot})
                test_accuracy = accuracy_test.eval(feed_dict={x: xs_test, y_: ys_test_onehot})
                print('%s:After %d training steps, loss = %g, accuracy = %g, validation accuracy=%g'% (datetime.now(), i,
                                                                    loss_value, train_accuracy, test_accuracy))

                # 保存当前的模型。这里给出了global_step参数，这样可以让每个被保存的文件名末尾加上训练的轮数，比如
                # 'model.ckpt-1000'表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys_onehot})
        coord.request_stop()
        coord.join(threads)
    writer.close()

# 下载数据集，并解压，展开到其默认位置
def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.mkdir(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(argv=None):
    maybe_download_and_extract()
    CIFAR10_train()

"""
    下述第一行代码表示如果当前是从其他模块调用的该模块程序，则不会运行main函数
    而如果就是直接运行的该模块程序，则会运行main函数。
    第二行的核心意思是执行程序中main函数，并解析命令行参数flag。
"""
if __name__ == '__main__':
    tf.app.run()

