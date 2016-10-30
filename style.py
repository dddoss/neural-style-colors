import params
import vgg
import Pillow as pil
import numpy as np

# Given two input images (one for content, the other for style), generates a novel image
# with the content of the first and the style of the second

def load_image(filepath):
    # use Pillow to load image into memory, convert to numpy array
    pass

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
    content_im = load_image(params.content_path)
    style_im = load_image(params.style_path)

    # Retrieve target activation weight matrices
    content_gram = vgg.gram_matrix(content_im)
    style_weights = vgg.activation_weights(style_im)

    generate_image(content_weights, style_gram)

if __name__=='__main__':
    main()
