# vgg.py: Given input images (as numpy arrays), run them through the pretrained vgg network and retreive activations

# Somehow statically load vgg into a form that can be held in-memory and quickly used to process images; some sort of global variable


# input_image is m*n numpy array; returns 3D numpy array with weights[i, j, l] cooresponding to filter i, position j, layer l
def weight_activations(input_image):
    # Take the input image, run it through the vgg network, return the weight activation matrix
    pass

# returns the Gram Matrix for the input image: gram[i, j, l] = sum_k (weights[i, k, l]*weights[j, k, l] )
def gram_matrix(input_image):
    pass
