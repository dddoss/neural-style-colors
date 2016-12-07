# color.py : given input images (as numpy arrays), perform luminance-only color transfer
import skimage.color
from skimage import color
from PIL import Image
import numpy as np
# many possible tactics to implement color, but main two are color histogram matching and luminance-only transfer

# First, Luminance-only
# For Luminance-only transfer:
# [not in this file] 0. Represent content and style images in YIQ color space
# [not in this file] 1. Extract luminance channels (Y) from style and content images
# [not in this file] 2. Use neural style transfer algorithm to make output luminance image
# 3. Combine I and Q channels with luminance image to make output image

def load_image(filepath):
    image = Image.open(filepath)
    image = image.convert(mode="RGB")
    rgb2yiq = (
        0.299, 0.587, 0.114, 0,
        0.596, -0.0274, -0.322, 0,
        0.211, -0.523, 0.312, 0 )
    out = image.convert("RGB", rgb2yiq)
#    out = skimage.color.rgb2yiq(image)  
    data = np.asarray(out, dtype='int32')
    return data

def combine_images(Im_y, Im_iq):
    combined_im = []
    for i in range(Im_y.shape[0]):
        combined_im_row = []
        for j in range(Im_y.shape[1]):
            combined_im_row.append([Im_y[i][j][0], Im_iq[i][j][1], Im_iq[i][j][2]])
        combined_im.append(combined_im_row)
    return np.asarray(combined_im)
a = load_image('test0.png')
b = load_image('test1.png')
c = combine_images(a, b)
max_c = 0
for i in range(128):
    for j in range(128):
        for k in range(2):
            max_c = max(max_c, c[i][j][k])
print('max: ', max_c)

im = Image.fromarray(np.uint8(c))
im.show()

# TODO: Luminance histogram matching before synthesizing luminance image
# TODO: Implement color histogram matching, possibly other methods
