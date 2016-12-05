import numpy as np

data = np.load('./vgg_model/vgg19_data.npy')
dataitem = data.item()
dataitem.pop('fc6', None)
dataitem.pop('fc7', None)
dataitem.pop('fc8', None)
for i in range(0,3):
  for j in range(0,3):
    temp = np.array(dataitem['conv1_1']['weights'][i, j, 0])
    dataitem['conv1_1']['weights'][i,j,0] = np.array(dataitem['conv1_1']['weights'][i,j,2])
    dataitem['conv1_1']['weights'][i,j,2] = temp
for key in dataitem.keys():
  if key[:4] == 'conv':
    dataitem[key]['weights'] = np.transpose(dataitem[key]['weights'], (1, 0, 2, 3))
np.save('./vgg_model/vgg19_data.npy', data)
