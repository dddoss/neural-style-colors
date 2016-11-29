"""params.py: default parameters for the neural style algorithm"""
#TODO: These should all be settable by command line flags to the actual script

content_path = "./images/mitart_lowres.jpg" # relative path of content input image
style_path = "./images/starrynight_lowres.jpg"     # relative path of style input image
output_path = "./images/output.png"         # relative path of output image


# Algorithm parameters
content_layer = 0                            # Which vgg layer to use for matchinging content
style_layers = [0, 2, 4, 8, 12]       # Which vgg layers to use for matching style
style_weights = [1.0/len(style_layers)]*len(style_layers) # How to relatively weight style layers; default to equal weights
iterations = 100;
alpha = 1.e-3                                 # alpha and beta are relative weighting of style vs content in output
beta = 1.
learning_rate = 1.0                         # arbitrarily picked
