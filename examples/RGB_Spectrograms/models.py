import keras
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, 
    BatchNormalization, LeakyReLU, AveragePooling1D, AveragePooling2D, Reshape
)
from keras.models import Model
from keras.regularizers import l2
import keras.ops as knp
import tensorflow as tf

from examples.RGB_Spectrograms.constants import NEW_INPUT_SHAPE, NEW_OUTPUT_SHAPE

def leaky_relu_conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), 
                          padding='same', regularization_strength=0.01, 
                          negative_slope=0.1, use_transpose=False, use_batch_norm=True):
    """
    A flexible convolutional block that can create either a Conv2D or Conv2DTranspose layer,
    followed by optional BatchNormalization and LeakyReLU activation.
    """
    ConvLayer = Conv2DTranspose if use_transpose else Conv2D

    x = ConvLayer(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation='linear',
        kernel_regularizer=l2(regularization_strength),
        bias_regularizer=l2(regularization_strength),
        use_bias=not use_batch_norm,
    )(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    x = LeakyReLU(negative_slope=negative_slope)(x)
    return x

def encoder_block(x, filters, pool_size=(2, 2), kernel_size=(3, 3), 
                  strides=(1, 1), regularization_strength=0.01):
    """
    A typical encoder block: convolution block(s) followed by max-pooling.
    """
    x = leaky_relu_conv_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, 
        regularization_strength=regularization_strength
    )
    p = MaxPooling2D(pool_size=pool_size)(x)
    return x, p

def decoder_block(x, skip_connection, filters, kernel_size=(3, 3), 
                  up_strides=(2, 2), regularization_strength=0.01):
    """
    A typical decoder block: upsampling (transpose conv), concatenation with skip connection, 
    followed by a convolution block.
    """
    x = leaky_relu_conv_block(
        x, filters=filters, kernel_size=kernel_size, strides=up_strides, 
        regularization_strength=regularization_strength, use_transpose=True
    )
    x = Concatenate()([x, skip_connection])
    return x

def segmentation_model(input_shape=NEW_INPUT_SHAPE, output_shape=NEW_OUTPUT_SHAPE, num_classes=4, from_logits=False):
    regularization_strength = 0.01
    input = Input(shape=input_shape)
    x = input

    # Apply Conv2d with strides to downsample frequencies
    pool_size = (2, 2)
    strides = (1, 1)
    kernel_horiz_0 = 19 * 2 * 12 # chosen such that 30 seconds corresponds to 1 kernel
    kernel_horiz_1 = 19 * 2 * 6
    # kernel_horiz = 7

    # experiment with more freq pixels
    kernel_vert = 7
    kernel_size_0 = (kernel_horiz_0, kernel_vert)
    kernel_size_1 = (kernel_horiz_1, kernel_vert)

    filters_base = 4
    filters_incr_ratio = 2
    current_filter = filters_base

    x0, p0 = encoder_block(x, filters=filters_base, pool_size=pool_size, kernel_size=kernel_size_0, strides=strides)

    current_filter *= filters_incr_ratio
    current_filter = int(current_filter)
    x1, p1 = encoder_block(p0, filters=current_filter, pool_size=pool_size, kernel_size=kernel_size_1, strides=strides,
                         regularization_strength=regularization_strength)
    current_filter *= filters_incr_ratio
    current_filter = int(current_filter)
    x2, p2 = encoder_block(p1, filters=current_filter, pool_size=pool_size, kernel_size=kernel_size_1, strides=strides,
                         regularization_strength=regularization_strength)
    current_filter *= filters_incr_ratio
    current_filter = int(current_filter)
    x3, p3 = encoder_block(p2, filters=current_filter, pool_size=pool_size, kernel_size=kernel_size_1, strides=strides,
                         regularization_strength=regularization_strength)
    # current_filter *= filters_incr_ratio
    # current_filter = int(current_filter)
    # x4, p4 = encoder_block(p3, filters=current_filter, pool_size=pool_size, kernel_size=kernel_size_1, strides=strides,
    #                      regularization_strength=regularization_strength)
    
    # Now apply 2 decoder blocks
    y = decoder_block(p3, x3, filters=current_filter, kernel_size=kernel_size_1, up_strides=pool_size,)
    current_filter //= filters_incr_ratio
    y = decoder_block(y, x2, filters=current_filter, kernel_size=kernel_size_1, up_strides=pool_size,)
    current_filter //= filters_incr_ratio
    y = decoder_block(y, x1, filters=current_filter, kernel_size=kernel_size_1, up_strides=pool_size,)
    y, q = encoder_block(y, filters=4, pool_size=pool_size, kernel_size=(3, 3), strides=strides,)
    # q = p4

    reshaped_inputs = Reshape((output_shape[0], -1))(q)



    # Apply a Conv1d layer to get num_classes
    final_activation = 'linear' if from_logits else 'softmax'
    outputs = keras.layers.Conv1D(
        num_classes, 
        1, 
        activation=final_activation,
        kernel_regularizer=l2(regularization_strength),
        bias_regularizer=l2(regularization_strength),
        use_bias=True,
        )(reshaped_inputs)


    model = Model(input, outputs)
    return model

def segmentation_model_big(input_shape=NEW_INPUT_SHAPE, num_classes=4, frequency_downsample=4, from_logits=False):
    inputs = Input(shape=input_shape)

    # Downsample frequencies first
    pooled_inputs = AveragePooling2D(pool_size=(1, frequency_downsample))(inputs)

    # Define filters in a single place
    filters = [2, 4, 8]  # This corresponds to [2**1, 2**2, 2**3]

    # Encoder
    c1, p1 = encoder_block(pooled_inputs, filters[0])   # Level 1
    c2, p2 = encoder_block(p1, filters[1])              # Level 2
    c3, p3 = encoder_block(p2, filters[2])              # Level 3

    # Further downsampling after the last encoder block
    # p3 = leaky_relu_conv_block(p3, filters=filters[0], kernel_size=(3, 3), strides=(2, 2))

    # Decoder
    # Up-sample and fuse with skip connections
    u1 = leaky_relu_conv_block(p3, filters=filters[2], kernel_size=(3, 3), strides=(2, 2), use_transpose=True)
    # Here, instead of another decoder_block, let's follow the pattern:
    # (If you need more symmetrical structure, consider adding symmetrical decoder blocks.)
    u2 = Concatenate()([u1, c3])
    u2 = leaky_relu_conv_block(u2, filters=filters[1], kernel_size=(3, 3), strides=(2, 2))

    final_rep = u2


    # Collapse spatial dimensions as needed.
    # Note: Adjusting the pooling to ensure the final shape matches your target.
    # It's unclear from the original code what exact temporal/frequency downsampling is intended.
    # You may need to adapt pool sizes based on your input dimensions and desired output shape.
    collapse = Reshape((2048, -1))(final_rep)

    # Average pool to halve the time dimension
    collapse = AveragePooling1D(pool_size=2)(collapse)

    # Reshape to (time_steps, features); time_steps and features inferred dynamically
    # Since final dimensions can vary, here we assume something like (batch, time, freq, channels).
    # Replace 1024 and -1 with dynamically inferred shapes if known. For demonstration:
    # Suppose final_rep shape is (batch, T, F, C), we want (batch, T, num_classes).
    # If we assume T = 1024 and F * C collapses into num_classes after a Dense layer:

    # Apply a Dense layer to get num_classes
    final_activation = 'linear' if from_logits else 'softmax'
    dense_layer = keras.layers.Dense(num_classes, activation=final_activation)
    outputs = keras.layers.TimeDistributed(dense_layer)(collapse)

    model = Model(inputs, outputs)
    return model
