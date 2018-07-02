# author:SuperDan
# _*_ coding: utf-8 _*_

import tensorflow as tf
import cifar10_input
import pylab

BATCH_SIZE = 128
data_dir = '../data-bin'

# 显示被裁减和归一化后的图片
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=BATCH_SIZE)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
print('类别：',label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()
