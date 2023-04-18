"""
Author: Van Thong Huynh
Affiliation: Dept. of AI Convergence, Chonnam Nat'l Univ.
"""

import keras
from keras.applications import resnet, resnet_v2
import keras.layers as layers
from official.vision.modeling.layers.nn_layers import SpatialPyramidPooling


class ResNetBackbone(keras.Model):
    def __init__(self, input_shape=(224, 224, 3), version=1, weights='imagenet', level='0', **kwargs):
        input_tensor = layers.Input(shape=input_shape)
        if version == 1:
            base = resnet.ResNet50(include_top=False, weights=weights, input_tensor=input_tensor, pooling='avg')
        elif version == 2:
            base = resnet_v2.ResNet50V2(include_top=False, weights=weights, input_tensor=input_tensor, pooling='avg')
        else:
            raise ('ResNetBackbone do not know version {}'.format(version))

        base.trainable = True

        last_output = base.output

        outputs = dict()

        if '0' in level:
            outputs.update({'feat_last': last_output})

        if '1' in level:
            conv2_out = base.get_layer('conv2_block3_out').output
            conv2_out = layers.GlobalAvgPool2D()(conv2_out)
            outputs.update({'feat_conv2': conv2_out})
        if '2' in level:
            conv3_out = base.get_layer('conv3_block4_out').output
            conv3_out = layers.GlobalAvgPool2D()(conv3_out)
            outputs.update({'feat_conv3': conv3_out})
        if '3' in level:
            conv4_out = base.get_layer('conv4_block6_out').output
            conv4_out = layers.GlobalAvgPool2D()(conv4_out)
            outputs.update({'feat_conv4': conv4_out})

        print('Backbone features: ', outputs.keys())
        super(ResNetBackbone, self).__init__(inputs=base.inputs,
                                             outputs=outputs,
                                             **kwargs)

        self._config_dict = {'input_shape': input_shape, 'version': version, 'weights': weights, 'level': level}

    def get_config(self):
        return self._config_dict
