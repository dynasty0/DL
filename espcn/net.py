import tensorflow as tf 
import numpy as np 

# def to_tanh(input):
#     return 2.0*(input/255.0)-1.0
 
# def from_tanh(input):
#     return 255.0*(input+1)/2.0 

def espcn(input,factor):
    x = input
    n,h,w,c = x.get_shape().as_list()
    x = tf.layers.conv2d(inputs = x,
                        filters = 64,
                        kernel_size = 3,
                        padding = 'same',
                        name = 'conv1')
    x = tf.tanh(x, name = 'tanh1')
    x = tf.layers.conv2d(inputs = x,
                        filters = 32,
                        kernel_size = 3,
                        padding = 'same',
                        name = 'conv2')
    x = tf.tanh(x, name = 'tanh2')
    x = tf.layers.conv2d(inputs = x,
                        filters = 3*factor*factor,
                        kernel_size = 3,
                        padding = 'same',
                        name = 'conv3')
    # subpixel
    x = tf.concat([tf.reshape(a,(-1,factor*h,factor,3)) for a in tf.split(x, w,axis = 2)],axis = 2)
    return x

def read_and_decode(filename,epochs):

	filename_queue = tf.train.string_input_producer([filename],num_epochs=epochs,shuffle=True)

	reader = tf.TFRecordReader()

	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
		serialized_example,
		features={
		'img_lr': tf.FixedLenFeature([], tf.string),
		'img_raw': tf.FixedLenFeature([], tf.string)
		}
	)
	img_raw = features['img_raw']
	img_raw = tf.decode_raw(img_raw, tf.uint8)
	img_raw = tf.reshape(img_raw, [96, 96, 3])
	img_raw = tf.cast(img_raw, tf.float32) * (2. / 255) - 1.
	img_lr = features['img_lr']
	img_lr = tf.decode_raw(img_lr, tf.uint8)
	img_lr = tf.reshape(img_lr, [32, 32, 3])
	img_lr = tf.cast(img_lr, tf.float32) * (2. / 255) - 1.
	return img_lr, img_raw