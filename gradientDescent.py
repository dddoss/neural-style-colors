import numpy as np
import random

#matrix style size is m x n in np array format
#matrix content size is c x d

#TODO: Add loss calculation is equal to content loss + style loss + variations  
#PASS IN ITERATIONS, LEARNING RATE, LOSS


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



