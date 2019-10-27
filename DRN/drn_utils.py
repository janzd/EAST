"""
Author: Shishir Jakati
Implementation Inspired By: https://github.com/fyu/drn
"""

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


def Conv3x3(in_filters, out_channels, stride=1, padding=1, dilation=1, downsample=None):

    conv3x3 = keras.models.Sequential()

    conv3x3.add(Input(shape=(None, None, in_filters)))
    conv3x3.add(CustomPaddingConv2D(out_channels, stride=stride, padding=padding, dilation=dilation))
    
    if downsample is not None:
        conv3x3.add(downsample)

    return conv3x3

def Downsample(in_filters, planes, expansion_term, stride=1):

    downsample = keras.models.Sequential()

    downsample.add(Conv2D(planes * expansion_term, kernel_size=(1,1), strides=(stride, stride)))
    downsample.add(BatchNormalization(planes * expansion_term))

    return downsample
