"""
Author: Shishir Jakati
"""

import numpy as np

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, ZeroPadding2D


class CustomPaddingConv2D():

    """
    Custom Layer Model Used For Custom Padding Dimension. Additionally, can specify some normal Conv2D params.
    """
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CustomPaddingConv2D, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def GetLayer(self):
        """
        Returns a Convolutional Layer over a Custom Padded Layer
        """
        layer = keras.layers.Lambda(lambda x: 
                    Conv2D(self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), dilation_rate=(self.dilation, self.dilation), strides=(self.stride, self.stride))(
                        ZeroPadding2D(padding=[[self.padding, self.padding], [self.padding, self.padding]])(x)
                    )
                )
        return layer
