import keras
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, LeakyReLU,
    AveragePooling2D, AveragePooling3D, Reshape
)
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

from examples.RGB_Spectrograms.constants import NEW_INPUT_SHAPE, NEW_OUTPUT_SHAPE


def leaky_relu_block(x, filters, kernel_size, strides, padding='same', regularization_strength=0.01, negative_slope=0.1):

    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        activation='linear',
        kernel_regularizer=l2(regularization_strength),
        bias_regularizer=l2(regularization_strength))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=negative_slope)(x)
    return x

def leaky_relu_transpose_block(x, filters, kernel_size, strides, padding='same', 
                               regularization_strength=0.01, negative_slope=0.1, 
                               use_batch_norm=True):
    x = Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        activation='linear',
        kernel_regularizer=l2(regularization_strength),
        bias_regularizer=l2(regularization_strength))(x)
    x = LeakyReLU(negative_slope=negative_slope)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    return x

import keras
from keras.layers import (
    Input, Conv2D, AveragePooling2D, BatchNormalization, LeakyReLU, Reshape
)
from keras.models import Model

def segmentation_model(input_shape=NEW_INPUT_SHAPE, num_classes=4, from_logits=False):
    inputs = Input(shape=input_shape)

    filters = [2 ** i for i in range(1, 4)]
    
    # Encoder
    c1 = leaky_relu_block(inputs, filters[0], (3, 3), (1, 1))
    p1 = MaxPooling2D((2, 2))(c1)  # Downsampling
    
    c2 = leaky_relu_block(p1, filters[1], (3, 3), (1, 1))
    p2 = MaxPooling2D((2, 2))(c2)  # Further Downsampling

    c3 = leaky_relu_block(p2, filters[2], (3, 3), (1, 1))
    p3 = MaxPooling2D((2, 2))(c3)  # Further Downsampling
    p3 = leaky_relu_block(p3, filters[0], (3, 3), (2, 2))

    # # Decoder
    # u1 = leaky_relu_transpose_block(p3, filters[2], (3, 3), (2, 2))
    
    # u2 = leaky_relu_transpose_block(u1, filters[1], (3, 3), (2, 2))
    # u2 = Concatenate()([u2, c2])  # Skip connection

    # u2 = leaky_relu_block(u2, filters[0], (3, 3), (2, 2))
    # u2 = leaky_relu_block(u2, filters[0], (3, 3), (2, 2))
    # u2 = leaky_relu_block(u2, filters[0], (3, 3), (2, 2))
    u2 = p3
    # # Collapse frequency axis
    collapse = AveragePooling2D(pool_size=(1, u2.shape[2] // 2))(u2)

    # # Downsample to match label shape
    # final_downsampling = AveragePooling2D(pool_size=(15, 1))(collapse)  # Downsample time (* -> 1024)
    
    # # Final dense layer for class probabilities
    final_activation = 'linear' if from_logits else 'softmax'
    # outputs = Conv2D(num_classes, (1, 1), activation=final_activation)(final_downsampling)  # (1024, 4, 1)
    # Remove leftover spatial dimensions (1024, 4)
    outputs = Reshape((1024, -1))(collapse)

    if outputs.shape[-1] != num_classes:
        dense_layer = keras.layers.Dense(num_classes, activation=final_activation)
        outputs = keras.layers.TimeDistributed(dense_layer)(outputs)
    
    model = Model(inputs, outputs)
    return model
