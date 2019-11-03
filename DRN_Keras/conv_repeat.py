"""
Author: Shishir Jakati

Inspired by implementation here: https://github.com/dmolony3/DRN/blob/master/DRN.py
"""

from keras.layers import Add, BatchNormalization, Conv2D, Activation


def conv_repeat(input_channels, channel_shape, strides, dilation, kernel, residual, name):
    """
        Returns a model which performs repeated convolutional operations with residual connections
        Mimics the basic block in original implementation
    """
    input_tensor = Input(shape=(None, None, input_channels))
    X = input_tensor
    if residual == 1:
        shortcut = X
        if strides[-1] != -1:
            shortcut = Conv2D(channel_shape, (kernel, kernel), strides=(strides[-1], strides[-1]), padding="same", name=name + "_shortcut")(shortcut)
        elif shortcut.shape[2] != kernel[-1]:
            shortcut = Conv2D(channel_shape, (kernel, kernel), strides=(strides[0], strides[0]), padding="same", name=name+"_shortcut")(shortcut)

    for i, stride in enumerate(strides):
        
        if dilation:
            X = Conv2D(channel_shape, (kernel, kernel), strides=(strides[i], strides[i]), dilation=(dilation, dilation), padding="same", name=name + "_" + str(i))(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
        
        else:
            X = Conv2D(channel_shape, (kernel, kernel), strides=(strides[i], strides[i]), padding="same", name=name + "_" + str(i))(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
            if residual == 1 and len(strides) - 1 == i:
                X = Add()([shortcut, X])

            X = Activation('relu')(X)

    return keras.models.Model(inputs=input_tensor, outputs=X)
