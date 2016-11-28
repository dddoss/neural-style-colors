import params
import numpy as np
import tensorflow as tf
import random

#matrix style size is mxnx3 in np array format
#matrix content size is c x d

def total_loss(sess, content_network, style_network, content_weight, style_grams):
	#Calculate content loss
	loss_content = content_loss(sess, content_network)

	#Calculate style loss - Eq. 5 from paper
	E = [style_loss(sess.run(style_network[i]), style_network[i], N, M) for i in style_layers]
    W = [w for w in style_weights]
    loss_style = sum([W[l] * E[l] for l in range(len(style_layers))])

	#TODO: Add denoising
	total_variation_loss = 0

	loss = alpha*loss_content+beta*loss_style #+total_variation_loss
	return loss

#Squared-Error Loss between two feature representations
def content_loss(sess, content_network):
	#p is original image, x is generated image
	p = sess.run(content_network[content_layer])
	f = content_network[content_layer]

	N = p.shape[3] #number of distinct filters
	M = p.shape[1]*p.shape[2] #height times width of each feature maps

	#Eq. 1 in Neural Algorithm of Artistic Style Paper
	loss_content = 0.5 * tf.reduce_sum(tf.pow(x - f, 2))
	return loss_content

def style_loss(a, g, N, M):
	#Create Gram matrix for original image
	at = tf.reshape(a, (M,N))
	A = tf.matmul(tf.transpose(at), at)
	#Create Gram matrix for generated image
	gt = tf.reshape(g, (M,N))
	G = tf.matmul(tf.transpose(gt), gt)

	#Eq. 4 from paper
	style_loss = (1/(4* N**2 * M**2 ))* tf.reduce_sum(tf.pow(G-A, 2))
	return style_loss

def optimization(loss, learning_rate, iterations):
	init_op = tf.initizalize_all_variables()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    optimal = None
    optimalLoss = float('inf')

    with tf.Session as sess:
    	sess.run(init_op)
    	for i in range(iterations):
    		final = (i == iterations-1)
    		sess.run(train_op)

    		if final:
    			current = loss.eval()
    			if current < best:
    				optimalLoss = current
    				optimal = image.eval()
