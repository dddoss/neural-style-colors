"""params.py: default parameters for the neural style algorithm"""
#TODO: These should all be settable by command line flags to the actual script

content_path = "./images/mitart_lowres.jpg" # relative path of content input image
style_path = "./images/starrynight.jpg"     # relative path of style input image
output_path = "./images/output/lowres"         # relative path of output image


# Algorithm parameters
content_layer = 9                            # Which vgg layer to use for matching content
style_layers = [0, 2, 4, 8, 12]
style_weights = [1.0/len(style_layers)]*len(style_layers) # How to relatively weight style layers; default to equal weights
iterations = 1000
checkpoint = 200
content_weight = 5                                 # alpha and beta are relative weighting of style vs content in output
style_weight = 100.
tv_weight = 100.
learning_rate = 2.0                         # arbitrarily picked
