import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
# from utils import *
from net import *


# The train sets path 
path = r'/home/dynasty/Documents/SRGAN/sr_32_3.tfrecords'
train_mode = True
epochs = 50
batch_size = 128

img_lr, img_raw = read_and_decode(path,epochs)

img_lr_batch, img_raw_batch = tf.train.shuffle_batch([img_lr, img_raw],
		batch_size=batch_size, num_threads = 4,capacity=2000,
		min_after_dequeue=1000)
	

img_sr_batch = espcn(img_lr_batch,factor=3)

mse = tf.reduce_mean(tf.squared_difference(tf.to_float(img_raw_batch), img_sr_batch))

train_op = tf.train.AdamOptimizer(0.0001).minimize(mse)


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
#session = tf.Session(config=config, ...)

with tf.Session(config=config) as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	sess.run(tf.local_variables_initializer())
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord = coord)
	i=0
	try:
		while not coord.should_stop():
			# Run training steps or whatever
			i=i+1
			sess.run(train_op)
			print('aaa{}'.format(i),sess.run(mse))
			if i%1000==0:
				save_path = saver.save(sess, 'sr.tfmodel')

	except tf.errors.OutOfRangeError:
		save_path = saver.save(sess, 'sr-final.tfmodel')
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
