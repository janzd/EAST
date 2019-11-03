"""
Author: Shishir Jakati

Inspired by implementation here: https://github.com/dmolony3/DRN/blob/master/DRN.py
"""


import keras
from conv_repeat import conv_repeat
from basic_block import basic_block
from keras.models import Model
from keras.layers import Conv2D, Input, BatchNormalization, Activation


def build_DRN26(input_tensor):
    
    if input_tensor is None:
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
    residual = 0
    channels = 512
    repeat = conv_repeat(input_channels=512, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer7")
    Y = repeat(Y)

    strides = [1, 1]
    dilation = None
    kernel_size = 3
    residual = 0
    channels = 512
    repeat = conv_repeat(input_channels=512, channel_shape=channels, strides=strides, dilation=dilation, kernel=kernel_size, residual=residual, name="layer8")
    Y = repeat(Y)


    return keras.models.Model(inputs=input_tensor, outputs=Y)


def build_DRN42(input_tensor):
    """
    Creates the DRNC_42 Model.
    Inputs:
        None
    Outputs:
        - model: DRNC_42 Keras Model
    """
    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3))

    ## input block
    y = Conv2D(16, (7,7), name='layer1_1')(input_tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')

    ## layer 1
    block = basic_block(16, 16)
    y = block(y)

    ## layer 2
    block = basic_block(16, 32, strides=2, residual=False)
    y = block(y)

    ## layer 3
    block = basic_block(32, 64, strides=2, residual=False)
    y = block(y)
    block = basic_block(64, 64, strides=2)
    y = block(y)
    y = block(y)

    ## layer 4
    block = basic_block(64, 128, strides=2, residual=False)
    y = block(y)
    block = basic_block(128, 128, strides=2)
    y = block(y)
    y = block(y)
    y = block(y)

    ## layer 5
    block = basic_block(128, 256, dilation=2, residual=False)
    y = block(y)
    block = basic_block(256, 256, dilation=2)
    y = block(y)
    y = block(y)
    y = block(y)
    y = block(y)
    y = block(y)

    ## layer 6
    block = basic_block(256, 512, dilation=4, residual=False)
    y = block(y)
    block = basic_block(512, 512, dilation=4)
    y = block(y)
    y = block(y)

    ## layer 7
    block = basic_block(512, 512, dilation=2, residual=False)
    y = block(y)

    ## layer 8
    block = basic_block(512, 512, residual=False)
    y = block(y)

    return keras.models.Model(inputs=input_tensor, outputs=y)
