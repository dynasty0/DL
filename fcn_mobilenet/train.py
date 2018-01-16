# from net import *
import tensorflow as tf
import os
import scipy.io
import numpy as np
from mobilenet import *
slim = tf.contrib.slim


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

# lr = 1e-4
sample_size = 2927

path = r'/home/dynasty/work/segmentation/segmentation.tfrecords'
train_mode = True
epochs = 100
batch_size = 8

img, label = read_and_decode(path,epochs)
img_batch, label_batch = tf.train.shuffle_batch([img, label],
		batch_size=batch_size, capacity=2000,
		min_after_dequeue=1000)

### mobilenet restore
probs = fcn2(img_batch)

# saver = tf.train.Saver()
variables_to_restore = slim.get_variables(scope="MobilenetV1")

saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/home/dynasty/Downloads/mobilenet_v1_1.0_224.ckpt")
  print("Model restored.")
  # print variables_to_restore


loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=probs,
                                                                      labels=tf.squeeze(tf.to_int32(label_batch), squeeze_dims=[3]),
                                                                      name="entropy")))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True																	  
# optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-3,global_step,decay_steps=5*sample_size/batch_size,decay_rate=0.9,staircase=True) 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.Session(config=config) as sess:
    i=0
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord = coord)
    
    saver = tf.train.Saver()
    try:
        while not coord.should_stop():
            # Run training steps or whatever
            i=i+1
            global_step = i
            sess.run(optimizer)
            lr = sess.run(learning_rate) 
            # sess.run(train_op)
            print('iter {} ,lr:{} '.format(i,lr),sess.run(loss))
            if i%1000==0:
                save_path = saver.save(sess, './fcn.tfmodel')
    except tf.errors.OutOfRangeError:
        # save_path = saver.save(sess, './fcn-final.tfmodel')
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
