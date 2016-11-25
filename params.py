"""params.py: default parameters for the neural style algorithm"""
#TODO: These should all be settable by command line flags to the actual script

content_path = "./images/mitart.jpg" # relative path of content input image
style_path = "./images/starrynight.jpg"     # relative path of style input image
output_path = "./images/output.png"         # relative path of output image


# Algorithm parameters
content_layer = -1                            # Which vgg layer to use for matchinging content
style_layers = [-1, -2, -3, -4, -5]       # Which vgg layer to use for matching style
style_weights = [1.0/len(style_layers)]*len(style_layers) # How to relatively weight style layers; default to equal weights
alpha = 0.5                                 # alpha ~ [0, 1]; relative weighting of style vs content in output

