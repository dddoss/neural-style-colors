import params
import numpy as np
import tensorflow as tf
import random

#matrix style size is mxnx3 in np array format
#matrix content size is c x d

def total_loss(content_acts, style_grams, out_acts, out_grams):
    #Calculate content loss
    loss_content = content_loss(content_acts, out_acts)

    #Calculate style loss - Eq. 5 from paper
    loss_style = style_loss(style_grams, out_grams)

    #TODO: Add denoising
    total_variation_loss = 0

    loss = params.alpha*loss_content+params.beta*loss_style #+total_variation_loss
    return loss

#Squared-Error Loss between two feature representations
def content_loss(content_acts, out_acts):
    # in is original, out is generated
    content_in = content_acts[params.content_layer]
    content_out = out_acts[params.content_layer]

    # loss = 0.5*sum((out_acts-in_acts))
    # Artistic Style, eq. 1
    loss = tf.nn.l2_loss(tf.reduce_sum(content_out-content_in))
    return loss

def style_loss(style_grams, out_grams):
    layer_losses = []
    for i, l in enumerate(params.style_layers):
        in_gram = style_grams[l]
        out_gram = out_grams[l]
        in_shape = in_gram.get_shape()
        print(in_shape)
        Nl = in_shape[0].value
        Ml = in_shape[1].value
        #Eq. 4 from paper
        layer_loss = (1/(4* Nl**2 * Ml**2 ))* tf.reduce_sum(tf.square(in_gram-out_gram))
        layer_losses.append(layer_loss * params.style_weights[i])

    style_loss = tf.add_n(layer_losses)
    return style_loss

def optimization(loss, sess):
    init_op = tf.initialize_all_variables()
    train_op = tf.train.AdamOptimizer(params.learning_rate).minimize(loss)

    optimal = None
    optimalLoss = float('inf')
    sess.run(init_op)
    for i in range(params.iterations):
        print('iteration '+str(i))
        sess.run(train_op)

        thisLoss = loss.eval()
        if thisLoss < optimalLoss:
            print('better loss: '+str(thisLoss))
            optimalLoss = thisLoss
            optimal = output.eval()

    return optimal
