#encoding=utf-8
#from mobilenet_util import *
import tensorflow as tf
import numpy as np
from util import *

NUM_OF_CLASSESS = 3
keep_prob = 0.5
def to_tanh(input):
    return 2.0*(input/255.0)-1.0
 
def from_tanh(input):
    return 255.0*(input+1)/2.0 

def read_and_decode(filename,epochs):

    filename_queue = tf.train.string_input_producer([filename],num_epochs=epochs,shuffle=True)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
        'img_orginal': tf.FixedLenFeature([], tf.string),
        'img_segmentation': tf.FixedLenFeature([], tf.string)
        }
    )
    img = features['img_orginal']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [250, 250, 3])
    img = tf.cast(img, tf.float32) * (2. / 255) - 1.
    label = features['img_segmentation']
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [250, 250, 1])
    return img, label

def fcn(image,weights=None): ### use bilinear
    net = vgg_net(image,weights=weights)
    conv_final_layer = net["conv5_3"]
    pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv6 = tf.layers.conv2d(pool5,4096,7,padding='SAME',name='conv6')
    relu6 = tf.nn.relu(conv6, name="relu6")
    relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

    conv7 = tf.layers.conv2d(relu_dropout6,4096,1,name="conv7")
    relu7 = tf.nn.relu(conv7, name="relu7")
    relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

    # now to upscale to actual image size
    deconv_shape1 = net["pool4"].get_shape()
    conv8 = tf.layers.conv2d(relu_dropout7,NUM_OF_CLASSESS,1,padding="SAME")
    conv_t1 = tf.image.resize_bilinear(conv8,deconv_shape1[1:3])
    tmp1 = tf.layers.conv2d(net["pool4"],NUM_OF_CLASSESS, 1 ,padding='SAME')
    fuse_1 = tf.add(conv_t1, tmp1, name="fuse_1")

    deconv_shape2 = net["pool3"].get_shape()
    conv_t2 = tf.image.resize_bilinear(fuse_1,deconv_shape2[1:3])
    tmp2 = tf.layers.conv2d(net["pool3"],NUM_OF_CLASSESS,1,padding="SAME")
    fuse_2 = tf.add(conv_t2, tmp2, name="fuse_2")

    shape = tf.shape(image)
    conv_t3 = tf.image.resize_bilinear(fuse_2,shape[1:3])
    return conv_t3



def fcn2(image,weights=None): ### use deconvolution
    net = vgg_net(image,weights=weights)
    conv_final_layer = net["conv5_3"]
    # print(conv_final_layer.get_shape().as_list())
    pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    W6 = weight_variable([7, 7, 512, 4096], name="W6")
    b6 = bias_variable([4096], name="b6")
    conv6 = conv2d_basic(pool5, W6, b6)
    relu6 = tf.nn.relu(conv6, name="relu6")
    relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

    W7 = weight_variable([1, 1, 4096, 4096], name="W7")
    b7 = bias_variable([4096], name="b7")
    conv7 = conv2d_basic(relu_dropout6, W7, b7)
    relu7 = tf.nn.relu(conv7, name="relu7")
    relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

    W8 = weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
    b8 = bias_variable([NUM_OF_CLASSESS], name="b8")
    conv8 = conv2d_basic(relu_dropout7, W8, b8)
    # now to upscale to actual image size
    deconv_shape1 = net["pool4"].get_shape()
    W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
    b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(net["pool4"]))
    fuse_1 = tf.add(conv_t1, net["pool4"], name="fuse_1")

    deconv_shape2 = net["pool3"].get_shape()
    W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
    b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net["pool3"]))
    fuse_2 = tf.add(conv_t2, net["pool3"], name="fuse_2")

    shape = tf.shape(image)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    W_t3 = weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
    b_t3 = bias_variable([NUM_OF_CLASSESS], name="b_t3")
    conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    # return tf.expand_dims(annotation_pred, dim=3), conv_t3
    return conv_t3


