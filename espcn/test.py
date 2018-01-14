#encoding=utf-8
import tensorflow as tf
import cv2
import numpy as np
import sys
import time
from net import *

#train_mode = True

time1 = time.time()
img2 = cv2.imread(sys.argv[1])
# size2 = [img2.shape[0]//4,img2.shape[1]//4]
# img2 = img2[img2.shape[0]//2-120:img2.shape[0]//2+120,img2.shape[1]//2-120:img2.shape[1]//2+120]
# cv2.imwrite('test_origin.jpg',img2)
# size2=[60,60]
img2 = tf.expand_dims(img2, 0)
# img2_lr = tf.image.resize_bicubic(img2, size2, name=None)


img_test = espcn(tf.to_float(img2)*(2.0/255.)-1.,3)
#init = tf.initialize_all_variables()
time2 = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
#sess.run(init)
saver.restore(sess, 'sr.tfmodel')

time3 = time.time()
tmp2 = tf.squeeze(img_test)
tmp2 = sess.run(tf.cast((tmp2+1.0)*255.0/2.0,tf.uint8))
time4 = time.time()
cv2.imwrite('img_sr.jpg',tmp2)
#cv2.imwrite('test_lr.jpg',sess.run(tf.squeeze(img2_lr)))

print('before session:',time2-time1)
print('start session time:', time3-time2)
print('run time:',time4-time3)