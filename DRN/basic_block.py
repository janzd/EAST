"""
Author: Shishir Jakati
Implementation Inspired By: https://github.com/fyu/drn
"""

import numpy as np

import keras
import keras.backend as K
import tensorflow as tf
import Conv3x3 from drn_utils
from keras.layers import BatchNormalization, Conv2D, ReLU, ZeroPadding2D, Input, Add


def BasicBlock(in_channels, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):

    
    x = Input(shape=(None, None, in_channels))

    ## this is where the rest of the layer logic will go
    ## name the internal logic as y
    y = Conv3x3(in_channels, )

    z = Add()([x, y])

    basic_block = keras.models.Model(inputs=x, outputs=z)
    return basic_block





class BasicBlock(keras.layers.Layer):

    def __init__(self, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):

        super(BasicBlock, self).__init__()

        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.dilation = dilation

    def call(self, x):
        
        ## set residual to identity
        residual = x

        ## conv1
        x = conv3x3(input_tensor,out_channels, 
                                stride=stride, padding=dilation[0], dilation=dilation[0])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        ## conv2
        x = conv3x3(x, out_channels,
                            stride=stride, padding=dilation[1], dilation=dilation[1])
        x = BatchNormalization()(x)

        if downsample is not None:
                residual = downsample(x)
        if residual:
            out += residual
        out = ReLU(out)

        return out
