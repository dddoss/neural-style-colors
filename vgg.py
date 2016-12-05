# vgg.py: Given input images (as numpy arrays), run them through the pretrained vgg network and retreive activations

from vgg_model import vgg19_code
import tensorflow as tf
import numpy as np
import scipy.misc
import random
import string

def load_image(filepath):
    image = scipy.misc.imread(filepath, mode='RGB').astype(np.float32)
    return image

def net(image, sess, scope=None):
    if scope==None:
        scope = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    with tf.variable_scope(scope):
        vgg_network = vgg19_code.VGG_ILSVRC_19_layers({'input': image}, trainable=False)
        vgg_network.load('./vgg_model/vgg19_data.npy', sess)
        return (vgg_network.get_act_mats(), vgg_network.get_grams())

# MEAN_PIXEL is a property of the VGG 19 network; it is sourced from the original paper
# The network weights have been transformed to match an RGB image, so we don't have to
# rotate the image channels here
MEAN_PIXEL = np.array([123.68, 116.779, 103.939]).astype(np.float32)
def preprocess(image):
    newimg = np.array(image-MEAN_PIXEL)
    return newimg

def postprocess(image):
    newimg = np.array(image)
    newimg = np.clip(newimg+MEAN_PIXEL, 0, 255).astype(np.uint8)
    return newimg
