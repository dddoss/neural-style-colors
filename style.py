import params
import vgg
import numpy as np
import gradientDescent as gd
import tensorflow as tf
import scipy.misc

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

    # Initialize the random noise image and build network for generating its activations
    initial = tf.random_normal(output_shape, dtype=tf.float32)*0.256
    output_var = tf.Variable(initial, dtype=tf.float32, name='output_img')
    out_acts, out_grams = vgg.net(tf.expand_dims(output_var, 0), sess, scope='output')

    # Build loss portion of graph
    loss = gd.total_loss(content_acts, style_grams, out_acts, out_grams, output_var)

    # Generate output image via back-propogation
    output_image = gd.optimization(loss, output_var, sess)
    return output_image

content_activations = None
style_grams = None
def main():
    global content_activations
    global style_grams
    print('Loading images')
    # Load images into memory as numpy arrays, preprocess to feed into VGG
    content_im = vgg.preprocess(vgg.load_image(params.content_path))
    style_im = vgg.preprocess(vgg.load_image(params.style_path))

    # Generate placeholders to get the activations for the inputs
    content_ph = tf.placeholder(tf.float32, shape=(1,)+content_im.shape, name='content_ph')
    style_ph = tf.placeholder(tf.float32, shape=(1,)+style_im.shape, name='style_ph')
    output_shape = content_im.shape

    # Retrieve activations for the given input images
    print('Building networks')
    with tf.Session() as sess:
        # Build the networks and run feedforward to get the activations for the inputs
        content_acts_tensor, _ = vgg.net(content_ph, sess, scope='content')
        _, style_grams_tensor = vgg.net(style_ph, sess, scope='style')
        feed_dict = {content_ph: np.array([content_im,]), style_ph: np.array([style_im,])}
        content_activations = sess.run(content_acts_tensor, feed_dict=feed_dict)
        style_grams = sess.run(style_grams_tensor, feed_dict=feed_dict)

        print('Generating image!')
        output = generate_image(sess, content_activations, style_grams, output_shape)
        scipy.misc.imsave(params.output_path+'.jpg', output)


if __name__=='__main__':
    main()
