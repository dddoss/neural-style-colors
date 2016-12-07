import params
import numpy as np
import tensorflow as tf
import random
from operator import mul
import scipy.misc
import vgg

#matrix style size is mxnx3 in np array format
#matrix content size is c x d

def total_loss(content_acts, style_grams, out_acts, out_grams, out_img):
    #Calculate content loss
    loss_content = content_loss(content_acts, out_acts)

    #Calculate style loss - Eq. 5 from paper
    loss_style = style_loss(style_grams, out_grams)

    # TV loss: remove high-frequency noise from image
    total_variation_loss = tv_loss(out_img)

    loss = params.content_weight*loss_content+params.style_weight*loss_style+params.tv_weight*total_variation_loss
    return loss

def tv_loss(out_img):
    (width, height, depth) = [out_img.get_shape()[i].value for i in range(0, 3)]

    # Take shifted slices from output
    out_left = tf.slice(out_img, [0, 0, 0], [width-1, -1, -1])
    out_right = tf.slice(out_img, [1, 0, 0], [width-1, -1, -1])
    out_up = tf.slice(out_img, [0, 0, 0], [-1, height-1, -1])
    out_down = tf.slice(out_img, [0, 1, 0], [-1, height-1, -1])
    # Minimize difference between adjacent pixel values
    tv_loss = 0.5*(tf.reduce_sum(tf.square(out_up-out_down))/tensor_size(out_up) +
                   tf.reduce_sum(tf.square(out_left-out_right))/tensor_size(out_left))
    return tv_loss

#Squared-Error Loss between two feature representations
def content_loss(content_acts, out_acts):
    # in is original, out is generated
    content_in = tf.constant(content_acts[params.content_layer])
    content_out = out_acts[params.content_layer]

    # Artistic Style, eq. 1
    # Normalize on size of output
    loss = tf.nn.l2_loss(content_out-content_in)/tensor_size(content_out)
    return loss

def style_loss(style_grams, out_grams):
    layer_losses = []
    for i, l in enumerate(params.style_layers):
        in_gram = tf.constant(style_grams[l])/style_grams[l].size
        out_gram = out_grams[l]/tensor_size(out_grams[l])

        #Eq. 4 from paper
        layer_loss = tf.nn.l2_loss(in_gram-out_gram)/tensor_size(out_grams[l])
        layer_losses.append(layer_loss * params.style_weights[i])

    style_loss = tf.add_n(layer_losses)
    return style_loss

def optimization(loss, output, sess):
    train_op = tf.train.AdamOptimizer(params.learning_rate).minimize(loss)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    optimal = output.eval()
    optimalLoss = float('inf')

    for i in range(params.iterations):
        if params.checkpoint and (i % params.checkpoint == 0):
           scipy.misc.imsave(params.output_path+'-check'+str(i)+'.jpg', vgg.postprocess(optimal)) 
            np.save(params.output_path+'-check'+str(i)+'.npy', optimal)
        print('iteration '+str(i))

        # Run one iteration of the training
        thisLoss, thisOut, _ = sess.run((loss, output, train_op))

        if thisLoss < optimalLoss:
            print('better loss: '+str(thisLoss))
            optimalLoss = thisLoss
            optimal = thisOut


    return vgg.postprocess(optimal)

def tensor_size(t):
    shape = t.get_shape()
    return reduce(mul, (shape[i].value for i in range(len(shape))), 1)
