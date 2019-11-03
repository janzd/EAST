"""
Author: Shishir Jakati

Inspired by implementation here: https://github.com/dmolony3/DRN/blob/master/DRN.py
"""

from keras.layers import Add, BatchNormalization, Conv2D, Activation

def basic_block(input_channels, output_channels, strides=1, dilation=1, kernel=3, residual=True):
    """
    Defines the basic block as used in the DRN. 

    Inputs:
        - input_channels: Channel size of input tensor
        - output_channels: Channel size of output tensor
        - strides: Stride length of convolutional filter
        - dilation: Dilation rate of convolutional filter
        - kernel: Kernel size of convolutional filter
        - residual: Flag indicating whether the residual connection is used or not
    
    Outputs:
        - model: BasicBlock Keras model
    """

    input_tensor = Input(shape=(None, None, input_channels))
    ## save input tensor for residual connection
    shortcut = input_tensor
    x = input_tensor

    ## first conv level
    y = Conv2D(output_channels, (kernel, kernel), strides=(strides, strides), dilation_rate=(dilation, dilation), padding="same")(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    ## second conv level
    y = Conv2D(output_channels, (kernel, kernel), strides=(strides, strides), dilation_rate=(dilation, dilation), padding="same")(y)
    y = BatchNormalization()(y)

    ## optional residual connection
    if residual:
        y = Add()([shortcut, y])
    
    y = Activation('relu')(y)

    ## return block
    return keras.models.Model(inputs=input_tensor, outputs=y)