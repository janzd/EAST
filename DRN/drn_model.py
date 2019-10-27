"""
Author: Shishir Jakati
Implementation Inspired By: https://github.com/fyu/drn
"""

import numpy as np

import keras
import keras.backend as K
import tensorflow as tf
from custom_padding import CustomPaddingConv2D
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.layers import (Activation, BatchNormalization, Conv2D, Dropout,
                          Input, Lambda, Layer, MaxPooling2D, ReLU, AveragePooling2D,
                          ZeroPadding2D, add, concatenate, multiply)
from keras.models import Model


def bottleneck(x, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):

    residual = x

    ## conv1
    x = Conv2D(out_channels, kernel_size=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    ## conv2
    x = CustomPaddingConv2D(out_channels, kernel_size=3, stride=stride, padding=dilation[1], dilation=dilation[1]).GetLayer()(x)
    x = BatchNormalization()(x)
    x = RelU(x)
    ## conv3
    x = Conv2D(out_channels * 4, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)

    if downsample is not None:
        residual = downsample(out_channels * 4)(x)

    out = x + residual
    out = ReLU(out)

    return out



def conv0_sequential(in_filters, out_filters):

    model = keras.models.Sequential()

    model.add(Conv2D(out_filters, (7, 7), strides=(1,1), input_shape=(None, None, in_filters)))
    model.add(BatchNormalization())
    model.add(ReLU())

    return model


class DRN(keras.Model):

    super(DRN, self).__init__(name='drn')

    def __init__(self, block, layers, num_classes=1, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False, out_middle=False, pool_size=28, arch='D'):

        self.in_channels = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':

            self.conv1 = CustomPaddingConv2D(self.in_channels, kernel_size=7, stride=1, padding=3, dilation=0)
            self.bn1 = BatchNormalization()
            
            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1
            )

            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2
            )
        elif arch == 'D':

            self.layer0 = conv0_sequential(3, channels[0])

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1
            )
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)
            )

        
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)

        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_layer=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = AveragePooling2D(pool_size=(pool_size, pool_size))
            self.fc = Conv2D(num_classes, kernel_size=(1,1), stride=(1,1), use_bias=True)


    def call():
        pass

    def _make_conv_layers(self, channels, convs, stride=1, padding=1, dilation=1):

        modules = []
        for i in range(convs):
            modules.extend([
                Conv2D(channels, kernel_size=(3,3),
                          stride=stride if i == 0 else 1,
                          padding=dilation, dilation=dilation),
                BatchNormalization(),
                ReLU()])
            self.in_channels = channels

        return keras.models.Sequential(layers=modules)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = downsample()

        layers = []
        layers.append(block(planes, stride, downsample, dilation=dilation if dilation==1 else (
            dilation // 2 if new_level else dilation, dilation)
        ), residual=residual)

        self.in_channels = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes, residual=residual, dilation=(dilation,dilation)))

        return keras.models.Sequential(layers=layers)
