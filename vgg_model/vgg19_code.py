from kaffe.tensorflow import Network
import tensorflow as tf

class VGG_ILSVRC_19_layers(Network):
    def setup(self):
	(self.feed('input')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv1_1')
             .relu(name='relu1_1')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv1_2')
             .relu(name='relu1_2')
             .avg_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv2_1')
             .relu(name='relu2_1')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv2_2')
             .relu(name='relu2_2')
             .avg_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv3_1')
             .relu(name='relu3_1')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv3_2')
             .relu(name='relu3_2')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv3_3')
             .relu(name='relu3_3')
             .conv(3, 3, 256, 1, 1, relu=False, name='conv3_4')
             .relu(name='relu3_4')
             .avg_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv4_1')
             .relu(name='relu4_1')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv4_2')
             .relu(name='relu4_2')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv4_3')
             .relu(name='relu4_3')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv4_4')
             .relu(name='relu4_4')
             .avg_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_1')
             .relu(name='relu5_1')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_2')
             .relu(name='relu5_2')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_3')
             .relu(name='relu5_3')
             .conv(3, 3, 512, 1, 1, relu=False, name='conv5_4')
             .relu(name='relu5_4')
             .avg_pool(2, 2, 2, 2, name='pool5'))
        self.convs = []
        for i in range(2):
            self.convs.append(self.layers['relu1_'+str(i+1)])
        for i in range(2):
            self.convs.append(self.layers['relu2_'+str(i+1)])
        for i in range(4):
            self.convs.append(self.layers['relu3_'+str(i+1)])
        for i in range(4):
            self.convs.append(self.layers['relu4_'+str(i+1)])
        for i in range(4):
            self.convs.append(self.layers['relu5_'+str(i+1)])

    def get_act_mats(self):
        acts_trans = [tf.transpose(self.convs[i], [3, 0, 1, 2]) for i in range(len(self.convs))]
        acts =  [tf.reshape(acts_trans[i], [acts_trans[i].get_shape()[0].value, -1]) for i in range(len(acts_trans))]
        return acts

    def get_grams(self):
        acts = self.get_act_mats()
        grams = [tf.matmul(acts[i], tf.transpose(acts[i])) for i in range(len(acts))]
        return grams
