import params
import vgg
import numpy as np
import gradientDescent as gd
from PIL import Image
import tensorflow as tf

# Given two input images (one for content, the other for style), generates a novel image
# with the content of the first and the style of the second

def generate_image(sess, content_acts, style_grams, output_shape):
    # Use TensorFlow to generate an image with the same gram matrix as content_gram
    # and the same weight activations as style_weights via gradient descent.

    # 1. Initialize output as random noise
    # 2. Calculate gram matrix and activations matrix for output
    # 3. Error is difference between given matrices and output matrices
    # 4. Gradient descent to improve error
    # 5. Repeat until some threshold is reached

    output_var = tf.Variable(tf.random_uniform(output_shape, minval=0, maxval=255, dtype=tf.float32, name='output_img'))
    out_acts, out_grams = vgg.vgg_variable(tf.expand_dims(output_var, 0), sess, scope='output')
    loss = gd.total_loss(content_acts, style_grams, out_acts, out_grams, output_var)
    output_image = gd.optimization(loss, output_var, sess)
    output_image = np.clip(output_image, 0, 255).astype('uint8')
    return output_image


def main():
    # Load images into memory as numpy arrays
    print('Loading images')
    content_im = vgg.load_image(params.content_path)
    style_im = vgg.load_image(params.style_path)
    output_shape = content_im.shape
    print(output_shape)

    # Retrieve activations for the given input images
    print('Building networks')
    with tf.Session() as sess:
        content_activations, _ = vgg.vgg_constant(content_im, sess, scope='content')
        _, style_grams = vgg.vgg_constant(style_im, sess, scope='style')

        print('Generating image!')
        output = generate_image(sess, content_activations, style_grams, output_shape)

        image = Image.fromarray(output)
        image.save(params.output_path+'.jpg')


if __name__=='__main__':
    main()
