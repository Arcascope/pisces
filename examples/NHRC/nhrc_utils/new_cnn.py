import tensorflow as tf
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation
)
from keras.models import Model

NEW_INPUT_SHAPE = (15360, 257, 3)

def segmentation_model(input_shape=NEW_INPUT_SHAPE, num_classes=4):
    inputs = Input(shape=input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # Downsampling
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # Further Downsampling
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # Bottleneck
    
    # Decoder
    u1 = UpSampling2D((2, 2))(p3)
    u1 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Concatenate()([u1, c2])  # Skip connection
    
    u2 = UpSampling2D((2, 2))(u1)
    u2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Concatenate()([u2, c1])  # Skip connection
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(u2)  # Class scores per pixel
    
    model = Model(inputs, outputs)
    return model
