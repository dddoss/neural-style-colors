import numpy as np

data = np.load('./vgg_model/vgg19_data.npy')
data[None][0].pop('fc6', None)
data[None][0].pop('fc7', None)
data[None][0].pop('fc8', None)
np.save('./vgg_model/vgg19_data.npy', data)
