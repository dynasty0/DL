from net import *
import tensorflow as tf
import os
import scipy.io
import numpy as np
#from mobilenet import *
from net import *
slim = tf.contrib.slim

# lr = 1e-4
sample_size = 2927

# path = r'/home/caodai/work/tf/segmentation_250x_250.tfrecords'
path = r'/home/dynasty/work/segmentation/segmentation.tfrecords'
train_mode = True
epochs = 100
batch_size = 8

img, label = read_and_decode(path,epochs)
img_batch, label_batch = tf.train.shuffle_batch([img, label],
		batch_size=batch_size, capacity=2000,
		min_after_dequeue=1000)

###case 1: used for mobilenet like network
# probs = fcn(img_batch)

###case 2: used for traditional network
w_path = '/home/dynasty/imagenet-vgg-verydeep-16.mat'
model_data = scipy.io.loadmat(w_path)
weights = np.squeeze(model_data['layers'])
probs = fcn(img_batch,weights=weights)

### case 3: mobilenet 
#probs = fcn2(img_batch)

# saver = tf.train.Saver()
#variables_to_restore = slim.get_variables(scope="MobilenetV1")


# variables_to_restore = slim.get_model_variables()
# variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
#saver = tf.train.Saver(variables_to_restore)

#with tf.Session() as sess:
  # Restore variables from disk.
 # saver.restore(sess, "/home/dynasty/Downloads/mobilenet_v1_1.0_224.ckpt")
 # print("Model restored.")
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
    # if i%10000==9999:
        # lr/=10
    # optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord = coord)
    
    saver = tf.train.Saver()
    # saver.restore(sess, 'mobilenet_v1_1.0_224.ckpt')
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
        #save_path = saver.save(sess, './fcn-final.tfmodel')
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
