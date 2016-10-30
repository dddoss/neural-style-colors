"""params.py: default parameters for the neural style algorithm"""
#TODO: These should all be settable by command line flags to the actual script

content_path = "./images/content_image.png" # relative path of content input image
style_path = "./images/style_image.png"     # relative path of style input image
output_path = "./images/output.png"         # relative path of output image


# Algorithm parameters
style_layer = -1                            # Which vgg layer to use for matchinging style
content_layers = [-1, -2, -3, -4, -5]       # Whcih vgg layer to use for matching content
alpha = 0.5                                 # alpha ~ [0, 1]; relative weighting of style vs content in output

