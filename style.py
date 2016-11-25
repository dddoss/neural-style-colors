import params
import vgg
import numpy as np

# Given two input images (one for content, the other for style), generates a novel image
# with the content of the first and the style of the second

def generate_image(content_gram, style_weights):
    # Use TensorFlow to generate an image with the same gram matrix as content_gram
    # and the same weight activations as style_weights via gradient descent.

    # 1. Initialize output as random noise
    # 2. Calculate gram matrix and activations matrix for output
    # 3. Error is difference between given matrices and output matrices
    # 4. Gradient descent to improve error
    # 5. Repeat until some threshold is reached
    pass

def main():
    # Load images into memory as numpy arrays
    print('Loading images')
    content_im = vgg.load_image(params.content_path)
    style_im = vgg.load_image(params.style_path)
    (content_height, content_width) = content_im.shape[0:2]
    (style_height, style_width) = style_im.shape[0:2]

    # Retrieve activations for the given input images
    print('Building networks')
    content_network = vgg.Vgg(content_height, content_width, scope='content')
    style_network = vgg.Vgg(style_height, style_width, scope='style')

    print('Getting activations')
    content_activ = content_network.get_activations(content_im)
    style_activ = style_network.get_activations(style_im)

    # Extract the subsets of weights and gram matrices used for training
    content_weight = content_activ.activations[params.content_layer]
    style_grams = [style_activ.grams[i] for i in params.style_layers]

    print('Generating image!')
    generate_image(content_weight, style_grams)

if __name__=='__main__':
    main()
