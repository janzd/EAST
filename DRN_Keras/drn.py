"""
Author: Shishir Jakati

Inspired by implementation here: https://github.com/dmolony3/DRN/blob/master/DRN.py
"""


import keras
from conv_repeat import conv_repeat
from keras.models import Model
from keras.layers import Conv2D, Input


def build_DRN26():
    
    input_tensor = Input(shape=(None, None, 3))

    Y = Conv2D(16, (7,7), name='layer1_1')(input_tensor)

    strides = [1, 2]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 16
    repeat = conv_repeat(input_channels=16, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer1_2")
    Y = repeat(Y)

    strides = [1, 2]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 32
    repeat = conv_repeat(input_channels=16, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer2")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 64
    repeat = conv_repeat(input_channels=32, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer3_1")
    Y = repeat(Y)
    
    strides=[1, 2]
    repeat = conv_repeat(input_channels=64, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer3_2")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 128
    repeat = conv_repeat(input_channels=64, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer4_1")
    Y = repeat(Y)
    repeat = conv_repeat(input_channels=128, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer4_2")
    Y = repeat(Y)
    

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 256
    repeat = conv_repeat(input_channels=128, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer5")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 512
    repeat = conv_repeat(input_channels=256, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer6")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 512
    repeat = conv_repeat(input_channels=512, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer7")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 1
    channels = 512
    repeat = conv_repeat(input_channels=512, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer8")
    Y = repeat(Y)


    return keras.models.Model(inputs=input_tensor, outputs=Y)