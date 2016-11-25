import vgg
import tensorflow as tf
import numpy as np

image = vgg.load_image('./images/starrynight.jpg')
(height, width) = image.shape[0:2]
network = vgg.Vgg(height, width)
activations = network.get_activations(image)
print("Shape of activation matrix for layer 0: "+str(activations.activations[0].shape))
print("Shape of gram matrix for layer 0: "+str(activations.grams[0].shape))
