# vgg.py: Given input images (as numpy arrays), run them through the pretrained vgg network and retreive activations

from vgg_model import vgg19_code
import tensorflow as tf
import numpy as np
import scipy.misc
import random
import string

# Somehow statically load vgg into a form that can be held in-memory and quickly used to process images; some sort of global variable
def vgg_constant(image, sess, scope=None):
    if scope==None:
        scope = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    with tf.variable_scope(scope):
        image_tensor = tf.expand_dims(tf.constant(image), 0)
        vgg_network = vgg19_code.VGG_ILSVRC_19_layers({'input': image_tensor}, trainable=False)
        vgg_network.load('./vgg_model/vgg19_data.npy', sess)
        return (vgg_network.get_act_mats(), vgg_network.get_grams())

def vgg_variable(variable, sess, scope=None):
    if scope==None:
        scope = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    with tf.variable_scope(scope):
        vgg_network = vgg19_code.VGG_ILSVRC_19_layers({'input': variable}, trainable=False)
        vgg_network.load('./vgg_model/vgg19_data.npy', sess)
        return (vgg_network.get_act_mats(), vgg_network.get_grams())

def load_image(filepath):
    image = scipy.misc.imread(filepath).astype(np.float32)
    return image
