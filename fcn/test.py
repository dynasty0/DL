#encoding=utf-8
import tensorflow as tf
import cv2
import numpy as np
import sys
import time
import scipy.io
from net import *

from PIL import Image as PILImage

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

time1 = time.time()
img2 = cv2.imread(sys.argv[1])
img2 = tf.expand_dims(img2, 0)


img_test = fcn(tf.to_float(img2)*(2.0/255)-1)
time2 = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True	
sess = tf.Session(config=config)

saver = tf.train.Saver()

saver.restore(sess, 'fcn.tfmodel')

tmp2 = tf.squeeze(img_test)
time3 = time.time()
score = sess.run(tmp2)
time4 = time.time()

out = score.argmax(axis=2).astype(np.uint8)
palette = get_palette(256)
output_im = PILImage.fromarray(out)
output_im.putpalette(palette)
output_im.save("output.png")


print('before session:',time2-time1)
print('start session time:', time3-time2)
print('run time:',time4-time3)
