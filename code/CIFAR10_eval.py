#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23 22:29
# @Author  : SuperDan
# @Site    : 
# @File    : CIFAR10_eval.py
# @Software: PyCharm
# 通过这个程序，可以在滑动平均模型上做测试

import tensorflow as tf

# 加载CIFAR10_inference和CIFAR10_train.py中定义的常量和函数
import CIFAR10_inference
import CIFAR10_train
import CIFAR10_input
import numpy as np
import time
import math

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('EVAL_INTERVAL_SECS', 200,
                            "How often to run the eval")

tf.app.flags.DEFINE_integer('num_examples', 10000,
                            "Numbers of examples to test")

def evaluate():
    """eval models of CIFAR10"""
    # build a new graph and make it default for eval
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            # read data for testing
            if not CIFAR10_train.data_dir:
                raise ValueError('Please supply a data_dir')

            images, labels = CIFAR10_input.inputs(eval_data=True, data_dir=CIFAR10_train.data_dir,
                                                  batch_size=CIFAR10_train.FLAGS.BATCH_SIZE)

        # 直接通过调用函数来计算前向传播的结果。因为测试时不关注正则化损失的值
        # 所以这里用于计算正则化损失的函数被设为None
        logits = CIFAR10_inference.inference(input_tensor=images, train=False, regularizer=None)

        with tf.name_scope('accuracy'):
            top_k_op = tf.nn.in_top_k(logits, labels, 1)


        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用滑动平均的函数来获取平均值了。
        variable_averages = tf.train.ExponentialMovingAverage(CIFAR10_train.FLAGS.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../log_test', tf.get_default_graph())

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            while True:
                # 下面这个函数会通过checkpoint文件自动找到目录中最新模型的文件名。
                ckpt = tf.train.get_checkpoint_state(CIFAR10_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                else:
                    print('No checkpoint file found')
                    return
                                # result = sess.run(merged, feed_dict={: xs_test, y_: ys_onehot})
                # writer.add_summary(result, global_step)

                num_iter = int(math.ceil(FLAGS.num_examples / CIFAR10_train.FLAGS.BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * CIFAR10_train.FLAGS.BATCH_SIZE
                i = 0
                while i < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    i += 1
                prediction = true_count / total_sample_count
                print('After %s steps training, validation accuracy=%g' % (global_step, prediction))
                summary = tf.Summary()
                summary.ParseFromString(sess.run(merged))
                summary.value.add(tag='accuracy_test', simple_value=prediction)
                writer.add_summary(summary, global_step)
                time.sleep(FLAGS.EVAL_INTERVAL_SECS)
            coord.request_stop()
            coord.join(threads)
            writer.close()


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
