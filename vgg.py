# vgg.py: Given input images (as numpy arrays), run them through the pretrained vgg network and retreive activations

from vgg_model import vgg19_code
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import string

class Vgg(object):

    # Somehow statically load vgg into a form that can be held in-memory and quickly used to process images; some sort of global variable
    def __init__(self, height, width, scope=None):
        if scope==None:
            scope = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

        self.sess = tf.Session()
        with tf.variable_scope(scope):
            self.image_ph = tf.placeholder(tf.float32, shape=(1, height, width, 3))
            self.vgg_network = vgg19_code.VGG_ILSVRC_19_layers({'input': self.image_ph})
            self.vgg_network.load('./vgg_model/vgg19_data.npy', self.sess)

    # input_image is m*n*3 numpy array;
    # returns VggActivations object complete with activations and gram matrices calculated
    def get_activations(self, input_image):
        return VggActivations(self.vgg_network.get_activations({self.image_ph: [input_image, ]}, self.sess))


class VggActivations(object):

    def __init__(self, activations_list):
        # list of numpy arrays: activations[l][i, j] is activation for layer l, filter i, position j
        self.activations = []

        # list of numpy arrays: grams[l][i, j] = sum_k (activations[l][i, k]*activations[l][j, k])
        self.grams = [] 

        for l in range(len(activations_list)):
            matrix = np.array(activations_list[l])
            matrix = np.rollaxis(matrix, 2) # moves filter axis to front
            matrix = matrix.reshape((matrix.shape[0], -1)) # flatten width x height into positions
            self.activations.append(matrix)
        for l in range(len(activations_list)):
            matrix = self.activations[l]
            gram = np.dot(matrix, matrix.T)
            self.grams.append(gram)
        
def load_image(filepath):
    image = Image.open(filepath)
    data = np.asarray(image, dtype='int32')
    return data
