import numpy as np
import tensorflow as tf

from mobilenet_v1 import *

slim = tf.contrib.slim

keep_prob = 0.5
NUM_CLASSES = 3

def weight_variable(shape, stddev=0.02, name=None):
	# print(shape)
	initial = tf.truncated_normal(shape, stddev=stddev)
	if name is None:
		return tf.Variable(initial)
	else:
		return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
	initial = tf.constant(0.0, shape=shape)
	if name is None:
		return tf.Variable(initial)
	else:
		return tf.get_variable(name, initializer=initial)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def fcn(inputs):
	# batch_size = 5
	# height, width = 224, 224
	# inputs = tf.random_uniform((batch_size, height, width, 3))
	net, end_points = mobilenet_v1_base(inputs)
	# net = slim.separable_conv2d(net, None, [7,7],
                                      # depth_multiplier=1,
                                      # stride=1,
                                      # rate=1,
                                      # normalizer_fn=slim.batch_norm,
                                      # scope='fc')
	net = slim.conv2d(net, 1024, [1, 1],
                            stride=1,
                            normalizer_fn=slim.batch_norm,
                            scope='fc')
	net = tf.nn.dropout(net, keep_prob=keep_prob)
	net = slim.conv2d(net, NUM_CLASSES, [1, 1], scope='conv14')
	# net = slim.conv2d_transpose(net,NUM_CLASSES,[4,4], stride = [2,2])
	
	pool4 = end_points['Conv2d_11_pointwise']
	pool4_m = slim.conv2d(pool4, NUM_CLASSES, [1, 1], scope='conv_pool4')
	deconv_shape1 = pool4_m.get_shape()
	W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_CLASSES], name="W_t1")
	b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
	conv_t1 = conv2d_transpose_strided(net, W_t1, b_t1, output_shape=tf.shape(pool4_m))
	fuse_1 = tf.add(conv_t1, pool4_m, name="fuse_1")
	# net = fuse_1
	
	pool3 = end_points['Conv2d_5_pointwise']
	pool3_m = slim.conv2d(pool3, NUM_CLASSES, [1, 1], scope='conv_pool3')
	deconv_shape2 = pool3_m.get_shape()
	W_t1 = weight_variable([4, 4, deconv_shape2[3].value, NUM_CLASSES], name="W_t2")
	b_t1 = bias_variable([deconv_shape1[3].value], name="b_t2")
	conv_t2 = conv2d_transpose_strided(fuse_1, W_t1, b_t1, output_shape=tf.shape(pool3_m))
	fuse_2 = tf.add(conv_t2, pool3_m, name="fuse_2")
	# net = fuse_2
	
	shape = tf.shape(inputs)
	deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_CLASSES])
	W_t3 = weight_variable([16, 16, NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
	b_t3 = bias_variable([NUM_CLASSES], name="b_t3")
	conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
	
	return conv_t3

def fcn2(inputs):
	# batch_size = 5
	# height, width = 224, 224
	# inputs = tf.random_uniform((batch_size, height, width, 3))
	net, end_points = mobilenet_v1_base(inputs)
	# net = slim.separable_conv2d(net, None, [7,7],
                                      # depth_multiplier=1,
                                      # stride=1,
                                      # rate=1,
                                      # normalizer_fn=slim.batch_norm,
                                      # scope='fc')
	net = slim.conv2d(net, 1024, [1, 1],
                            stride=1,
                            normalizer_fn=slim.batch_norm,
                            scope='fc')
	net = tf.nn.dropout(net, keep_prob=keep_prob)
	net = slim.conv2d(net, NUM_CLASSES, [1, 1], scope='conv14')
	# net = slim.conv2d_transpose(net,NUM_CLASSES,[4,4], stride = [2,2])
	
	pool4 = end_points['Conv2d_11_pointwise']
	pool4_m = slim.conv2d(pool4, NUM_CLASSES, [1, 1], scope='conv_pool4')
	deconv_shape1 = pool4_m.get_shape()
	#### use deconv
	# W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_CLASSES], name="W_t1")
	# b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
	# conv_t1 = conv2d_transpose_strided(net, W_t1, b_t1, output_shape=tf.shape(pool4_m))
	#### use bilinear
	conv_t1 = tf.image.resize_bilinear(net,deconv_shape1[1:3])
	fuse_1 = tf.add(conv_t1, pool4_m, name="fuse_1")
	# net = fuse_1
	
	pool3 = end_points['Conv2d_5_pointwise']
	pool3_m = slim.conv2d(pool3, NUM_CLASSES, [1, 1], scope='conv_pool3')
	deconv_shape2 = pool3_m.get_shape()
	#### use deconv
	# W_t1 = weight_variable([4, 4, deconv_shape2[3].value, NUM_CLASSES], name="W_t2")
	# b_t1 = bias_variable([deconv_shape1[3].value], name="b_t2")
	# conv_t2 = conv2d_transpose_strided(fuse_1, W_t1, b_t1, output_shape=tf.shape(pool3_m))
	conv_t2 = tf.image.resize_bilinear(fuse_1,deconv_shape2[1:3])
	fuse_2 = tf.add(conv_t2, pool3_m, name="fuse_2")
	# net = fuse_2
	
	shape = tf.shape(inputs)
	deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_CLASSES])
	#### use deconv
	# W_t3 = weight_variable([16, 16, NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
	# b_t3 = bias_variable([NUM_CLASSES], name="b_t3")
	# conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
	#### use bilinear
	conv_t3 = tf.image.resize_bilinear(fuse_2,deconv_shape3[1:3])
	
	return conv_t3
	
# if __name__ == '__main__':
	# fcn()