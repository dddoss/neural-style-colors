import vgg
import numpy as np

# load the image as a numpy array
image = vgg.load_image('./images/starrynight.jpg')

# Generate the TensorFlow VGG network with the given height and width
# For the actual style transfer, we will make two instances of vgg.Vgg:
# One for the content and generated images, one for the style input
(height, width) = image.shape[0:2]
network = vgg.Vgg(height, width)

# Actually get the activations for the given image
activations = network.get_activations(image)

# Prove that it worked--the matrices are shaped correctly!
print("Shape of activation matrix for layer 0: "+str(activations.activations[0].shape))
print("Shape of gram matrix for layer 0: "+str(activations.grams[0].shape))
