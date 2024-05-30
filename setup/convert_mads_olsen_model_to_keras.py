"""Creates a saved Keras model used in the `MOResUNetPretrained` classifier

This script gives self-contained code to take the saved `.h5` model weights from Mads Olsen's group and convert them to a saved Keras model. This makes it much more portable, since we aren't trying to train this model further.

All code and scientific credit goes to Mads Olsen and his group, see [their repo](https://github.com/MADSOLSEN/SleepStagePrediction). This is just copy-pasting as much as is needed here.

The saved `.h5` weights at `pisces/cached_models/mo_model-best.h5` are copied from that repo, as well. 

"""
import os
import numpy as np
import pkg_resources
from dataclasses import dataclass
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras_cv.layers import StochasticDepth
from tensorflow.keras.regularizers import l2

print("Converting Mads Olsen model to Keras...")

def ResUNet(input_shape,
            num_classes,
            num_outputs,
            depth=None,
            init_filter_num=8,
            filter_increment_factor=2 ** (1 / 3),
            kernel_size=(16, 1),
            max_pool_size=(2, 1),
            activation='gelu',
            output_layer='sigmoid',
            weight_decay=0.0,
            residual=False,
            stochastic_depth=False,
            data_format = 'channels_last') -> Model:


    if depth is None:
        depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0])

    x_input = layers.Input(shape=tuple(input_shape))
    x = x_input

    # zero-pad:
    zeros_to_add = int(2 ** (np.ceil(np.log2(input_shape[0]))) - input_shape[0])
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.ZeroPadding2D(padding=(zeros_to_add // 2, 0))(x)

    # preallocation
    features = init_filter_num
    skips = []
    features_list = []
    kernel_size_list = []
    max_pool_size_list = []


    # Encoder
    # ========================================================
    for i in range(depth):

        # append lists of variables:
        features_list.append(features)
        kernel_size_list.append(kernel_size)
        max_pool_size_list.append(max_pool_size)

        # Feature extractor
        x = conv_block(x=x, features=int(features), kernel_size=kernel_size, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)
        skips.append(x)
        features *= filter_increment_factor

        # Reshape output to subsequent layer:
        x = layers.Conv2D(int(features),
                          kernel_size=max_pool_size,
                          activation=None,
                          padding='same',
                          strides=max_pool_size,
                          data_format=data_format,
                          kernel_regularizer=l2(weight_decay),
                          bias_regularizer=l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)

        # update kernel_size and max_pool_size
        kernel_size = [min(ks, x_dim) for ks, x_dim in zip(kernel_size, x.shape[1:3])]
        if x.shape[2] / max_pool_size[1] < 1:
            max_pool_size = (max_pool_size[0], 1)


    # Middel part
    # ========================================================
    x = conv_block(x=x, features=int(features), kernel_size=kernel_size, activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual)


    # Decoder
    # ========================================================
    features_list = [int(f) for f in features_list]
    for count, i in enumerate(reversed(range(depth))):

        # upsample and convolve
        x = layers.Conv2DTranspose(features_list[i],
                                   kernel_size=[int(mp) for mp in max_pool_size_list[i]],
                                   strides=[int(mp) for mp in max_pool_size_list[i]],
                                   padding='same',
                                   activation=None,
                                   data_format=data_format)(x)
        x = layers.BatchNormalization()(x)

        # concatenate with layer from encoder with same dimensionality
        x = layers.concatenate([skips[i], x], axis=3)

        # feature extractor
        x = conv_block(x=x, features=features_list[i], kernel_size=kernel_size_list[i], activation=activation,
                       data_format=data_format, weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth)


    # Cut-off zero-padded segment:
    if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
        x = layers.Lambda(lambda z: z[:, zeros_to_add // 2: - zeros_to_add // 2, :, :])(x)

    # reshape
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    # non-linear activation:
    x = layers.Conv1D(filters=init_filter_num ,
                      kernel_size=1,
                      padding='same',
                      activation=activation,
                      kernel_regularizer=l2(weight_decay),
                      bias_regularizer=l2(weight_decay)
                      )(x)

    if input_shape[0] // num_outputs > 0:
        x = layers.AveragePooling1D(pool_size=input_shape[0] // num_outputs)(x)


    # non-linear activation:
    x = layers.Conv1D(filters=num_classes,
                  kernel_size=1,
                  padding='same',
                  activation=activation,
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay)
                  )(x)


    # Classification
    # ========================================================
    x = layers.Dense(units=num_classes,
                     activation=output_layer)(x)

    return Model(inputs=x_input, outputs=x)


def conv_block(x, features, kernel_size, data_format='channels_last', weight_decay=0.0,
               residual=True, stochastic_depth=True, activation='gelu'):

    # feature extractor
    x_ = layers.Conv2D(int(features),
                       kernel_size=kernel_size,
                       activation=activation,
                       padding='same',
                       data_format=data_format,
                       kernel_regularizer=l2(weight_decay),
                       bias_regularizer=l2(weight_decay))(x)
    x_ = layers.BatchNormalization()(x_)

    if residual:
        if x.shape[-1] != x_.shape[-1]:
            x = layers.Conv2D(int(features),
                              kernel_size=(1, 1),
                              activation=None,
                              padding='same',
                              data_format=data_format,
                              kernel_regularizer=l2(weight_decay),
                              bias_regularizer=l2(weight_decay))(x)
        if stochastic_depth:
            return StochasticDepth(survival_probability=0.9)([x, x_])
        else:
            return layers.Add()([x, x_])
    else:
        return x_


def determine_depth(temporal_shape, temporal_max_pool_size):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= round(temporal_max_pool_size)
    depth -= 1
    return depth


events = ['wake', 'light', 'deep', 'rem']
events_format = [
    {
        'name': 'wake', 
        'h5_path': 'wake',
        'probability': 1 / len(events)
    },
    {
        'name': 'light', 
        'h5_path': 'light',
        'probability': 1 / len(events)
    },
    {
        'name': 'deep', 
        'h5_path': 'deep',
        'probability': 1 / len(events)
    },
    {
        'name': 'rem', 
        'h5_path': 'rem',
        'probability': 1 / len(events)
    }
]
signals_format =  {
    "ACC_merge_fft_spec": {
        "add": True,
        "fs_post": 0.5,
        "h5_path": "acc_signal",
        "dimensions": [
            32,
            1
        ],
        "channel_idx": [
            0,
            1,
            2
        ],
        "preprocessing": [
            {
                "args": {
                    "window_size": 30
                },
                "type": "median"
            },
            {
                "args": {
                    "iqr_window": 300,
                    "median_window": 300
                },
                "type": "iqr_normalization_adaptive"
            },
            {
                "args": {
                    "threshold": 20
                },
                "type": "clip_by_iqr"
            },
            {
                "args": {
                    "nfft": 512,
                    "f_max": 6,
                    "f_min": 0,
                    "f_sub": 3,
                    "window": 320,
                    "noverlap": 256
                },
                "type": "cal_psd"
            }
        ],
        "transformations": {
            "freq_mask": {},
            "time_mask": {},
            "image_translation": {}
        },
        "batch_normalization": {}
    },
    "PPG_fft_spec": {
        "add": False,
        "fs_post": 0.5,
        "h5_path": "ppg_signal",
        "dimensions": [
            32,
            1
        ],
        "channel_idx": [
            0
        ],
        "preprocessing": [
            {
                "args": {},
                "type": "zscore"
            },
            {
                "args": {},
                "type": "change_PPG_direction"
            },
            {
                "args": {
                    "iqr_window": 301,
                    "median_window": 301
                },
                "type": "iqr_normalization_adaptive"
            },
            {
                "args": {
                    "threshold": 20
                },
                "type": "clip_by_iqr"
            },
            {
                "args": {
                    "nfft": 512,
                    "f_max": 2.1,
                    "f_min": 0.1,
                    "f_sub": 1,
                    "window": 320,
                    "noverlap": 256
                },
                "type": "cal_psd"
            }
        ],
        "transformations": {
            "freq_mask": {},
            "time_mask": {},
            "image_translation": {}
        },
        "batch_normalization": {}
    }
}
data_directory = '.'

dataset_params = {
    "h5_directory": data_directory, 
    "signals_format": signals_format,
    "window": 30 * 2 ** 10, 
    "number_of_channels": len(signals_format), 
    "events_format": events_format,
    "prediction_resolution": 30,
    "overlap": 0.25,
    "minimum_overlap": 0.1,
    "batch_size": 2,
    "cache_data": True,
    "n_jobs": 4,
    "use_mask": True,
    "load_signal_in_RAM": True
}

@dataclass
class DSLite:
    fsTime: float
    nSpace: int
    nChannels: int
    window: int
    prediction_resolution: int

    def __init__(self, h5_directory, signals_format, window, overlap, batch_size, minimum_overlap, events_format, number_of_channels, prediction_resolution, load_signal_in_RAM, use_mask, cache_data, n_jobs):
# datasets
        self.h5_directory = h5_directory

        # signal modalities
        self.signals_format = signals_format
        self.window = window
        self.number_of_channels = number_of_channels
        self.prediction_resolution = prediction_resolution
        self.overlap = overlap
        self.batch_size = batch_size
        self.predictions_per_window = window // prediction_resolution
        self.nChannels = sum([sf['dimensions'][-1] for sf in signals_format.values()])
        self.nSpace = [sf['dimensions'][0] for sf in signals_format.values()][0] # assumes same space resolution
        self.fsTime = [sf['fs_post'] for sf in signals_format.values()][0] # assumes same temporal resolution

        # events
        self.events_format = events_format
        self.minimum_overlap = minimum_overlap
        self.number_of_events = len(events_format)
        self.number_of_classes = len(events_format)
        self.event_probabilities = [event['probability'] for event in events_format]
        self.event_labels = [event['name'] for event in events_format]
        assert sum(self.event_probabilities) <= 1

        # training
        self.load_signal_in_RAM = load_signal_in_RAM
        self.use_mask = use_mask


ds_train = DSLite(**dataset_params)

# model creation
model_params = {
    'input_shape': [int(ds_train.fsTime * ds_train.window), ds_train.nSpace, ds_train.nChannels], 
    'num_classes': len(events),
    'num_outputs': ds_train.window // ds_train.prediction_resolution,
    'depth': 9,
    'init_filter_num': 16,
    'filter_increment_factor': 2 ** (1 / 3),
    'max_pool_size': (2, 2),
    'kernel_size': (16, 3)
}

resunet = ResUNet(**model_params)
# Get the absolute path of the file
file_path = pkg_resources.resource_filename('pisces', 'cached_models/mo_model-best.h5')
file_path = file_path.replace(os.sep, '/') # OS independent path
# Load the weights
resunet.load_weights(filepath=file_path)
base_path = file_path.rsplit('/', 1)[0]
resunet.save(f'{base_path}/mo_resunet.keras')
print("Model saved as mo_resunet.keras")