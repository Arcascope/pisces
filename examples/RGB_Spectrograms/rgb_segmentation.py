import json
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#[Compiling module a_inference_one_step_on_data_58306__.69169] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
os.environ['XLA_FLAGS'] = '--xla_dump_to=./xla_dump'

# Use jax backend
# on macOS, this is one of the better out-of-the-box GPU options
# we have to do this first, before importing Keras ANYWHERE (including in pisces/other modules)
# So ignore the warnings about imports below this line
# pylint: disable=wrong-import-position,wrong-import-order
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from examples.RGB_Spectrograms.losses import MaskedTemporalCategoricalCrossEntropy
from examples.RGB_Spectrograms.plotting import debug_plot, overlay_channels_fixed
import sys
from pathlib import Path
from typing import List

from examples.RGB_Spectrograms.training import sparse_categorical_focal, sparse_kl_divergence



local_dir = Path(__file__).resolve().parent
print("local_dir: ", local_dir)
sys.path.append(str(local_dir.parent.joinpath('NHRC')))
from examples.RGB_Spectrograms.preprocessing import PreparedDataRGB, big_specgram_process, prepare_data
from examples.RGB_Spectrograms.channel_permuter import PermutationDataGenerator, Random3DRotationGenerator
from examples.RGB_Spectrograms.models import segmentation_model
from examples.RGB_Spectrograms.constants import NEW_INPUT_SHAPE, N_OUTPUT_EPOCHS
from dataclasses import dataclass
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
# import torch
from scipy.special import expit, softmax


import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import keras.ops as K


from src.constants import ACC_HZ
from pisces.metrics import WASAMetric, wasa_metric
from src.preprocess_and_save import do_preprocessing
from nhrc_utils.analysis import stages_map

def rgb_gather_reshape(data_bundle: PreparedDataRGB, train_idx_tensor: np.array, input_shape: tuple = NEW_INPUT_SHAPE, output_shape: tuple = (N_OUTPUT_EPOCHS,)) -> tuple | None:
    input_shape_stack = (-1, *input_shape)
    output_shape_stack = (-1, *output_shape)

    train_data = data_bundle.spectrograms[train_idx_tensor].reshape(input_shape_stack)
    train_labels = data_bundle.labels[train_idx_tensor].reshape(output_shape_stack)
    train_sample_weights = data_bundle.weights[train_idx_tensor].reshape(output_shape_stack)
    
    return train_data, train_labels, train_sample_weights

def rgb_path_name(key) -> str:
    os.makedirs("./saved_models", exist_ok=True)
    return f"./saved_models/rgb_{key}_{ACC_HZ}.keras"

def channelwise_mean(x, axes=[1, 2]):
    for axis in axes:
        x = np.mean(x, axis=axis, keepdims=True)
    return x

def channelwise_std(x, axes=[1, 2]):
    for axis in axes:
        x = np.std(x, axis=axis, keepdims=True)
    return x

def train_rgb_cnn(static_keys, static_data_bundle, hybrid_data_bundle, fit_callbacks: list = [], max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1, use_logits = False, n_classes=4):
    split_maker = LeaveOneOut()

    training_results = []
    cnn_predictors = []

    print(f"Training RGB CNN models...")
    WASA_PERCENT = 95
    WASA_FRAC = WASA_PERCENT / 100
    os.makedirs("./saved_outputs", exist_ok=True)

    def make_segmenter():
        return segmentation_model(num_classes=n_classes, from_logits=use_logits)
    
    print(make_segmenter().summary())

    # Split the data into training and testing sets
    for k_train, k_test in tqdm(split_maker.split(static_keys), desc="Next split", total=len(static_keys)):
        log_dir_cnn = f"./logs/rgb_cnn_{static_keys[k_test[0]]}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # Configure TensorBoard callback
        cnn_tensorboard_callback = TensorBoard(
            log_dir=log_dir_cnn, histogram_freq=1)
        if (max_splits > 0) and (len(training_results) >= max_splits):
                break
        # Convert indices to tensors
        train_idx_tensor = np.array(k_train)
        test_idx_tensor = np.array(k_test)

        # network instance to be trained
        cnn = make_segmenter()

        # Get the data
        train_data, train_labels, train_sample_weights = rgb_gather_reshape(
            static_data_bundle, train_idx_tensor)
        test_data, test_labels, test_sample_weights = rgb_gather_reshape(
            static_data_bundle, test_idx_tensor)

        # Train the model on the training set
        output = cnn.layers[-1].output
        print("output shape: ", output.shape)

        print("train label shape: ", train_labels.shape)
        print("test label shape: ", test_labels.shape)
        print("unique train labels: ", np.unique(train_labels))
        n_wake = float(np.sum(train_labels == 0))
        n_sleep = float(np.sum(train_labels > 0))

        n_total = n_wake + n_sleep
        class_weights = {
            -1: 0.0,
            0: (1 / n_wake) * (n_total / 2.0),
            1: (1 / n_sleep) * (n_total / 2.0)
        }
        print("class weights: ", json.dumps(class_weights, indent=2 ))

        # one-hot encode the labels
        train_labels = train_labels
        test_labels = test_labels
        # train_labels = keras.utils.to_categorical(
        #     train_labels,
        #     num_classes=n_classes)
        # test_labels = keras.utils.to_categorical(
        #     test_labels,
        #     num_classes=n_classes)

        temporal_ce = MaskedTemporalCategoricalCrossEntropy(n_classes=n_classes, sparse=True, from_logits=use_logits)

        cnn.compile(
            loss=temporal_ce,
            # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=use_logits),
            # loss=keras.losses.CategoricalFocalCrossentropy(from_logits=use_logits),
            optimizer=keras.optimizers.AdamW(learning_rate=lr),
            metrics=[temporal_ce],
            # weighted_metrics=[
            #     temporal_ce,
            #     # keras.losses.CategoricalFocalCrossentropy(from_logits=use_logits),
            # ]
        )

        # standardize the data
        train_specgram_mean = channelwise_mean(train_data)
        train_specgram_std = channelwise_std(train_data)
        train_data = (train_data - train_specgram_mean) / train_specgram_std

        # do this separately, to hide the difference of means from NN.
        # i.e. both inputs have mean 0 and std 1
        test_specgram_mean = channelwise_mean(test_data)
        test_specgram_std = channelwise_std(test_data)
        test_data = (test_data - test_specgram_mean) / test_specgram_std

        # channel_shuffler = PermutationDataGenerator(train_data, train_labels, sample_weights=train_sample_weights, batch_size=batch_size) 


        training_results.append(cnn.fit(
            train_data, train_labels,
            # channel_shuffler,
            # sample_weight=train_sample_weights,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            # validation_data=(test_data, test_labels, test_sample_weights),
            batch_size=batch_size,
            callbacks=[cnn_tensorboard_callback, *fit_callbacks]
        ))

        cnn_predictors.append(cnn)


        # Use cnn to predict probabilities
        # Rescale so everything doesn't get set to 0 or 1 in the expit call
        scalar = 10000.0 if use_logits else 1.0
        test_labels = test_labels[0]
        test_sample_weights = test_sample_weights[0]
        test_prediction_raw = cnn.predict(test_data) / scalar
        test_probabilities = softmax(test_prediction_raw, axis=-1) if use_logits else test_prediction_raw
        print("Plotting predictions")
        debug_plot(
            test_probabilities, 
            test_data, 
            weights=test_sample_weights,
            saveto=f"./saved_outputs/{static_keys[k_test[0]]}_cnn_pred_static_{ACC_HZ}.png")
        

        try:
            p_sleep = 1 - test_probabilities[:, 0]
            wasa, threshold = wasa_metric(
                labels=test_labels,
                predictions=p_sleep,
                weights=test_sample_weights)
            print(f"WASA{WASA_PERCENT}: {wasa.wake_accuracy:.4f}")
            test_pred_path = (static_keys[k_test[0]]) + \
                f"_cnn_pred_static_{ACC_HZ}.npy"
            test_pred = p_sleep > threshold
            np.save("./saved_outputs/" + test_pred_path, test_pred)
        except Exception as e:
            print(f"Error computing WASA: {e}")

        # save the trained model weights
        cnn_path = rgb_path_name(static_keys[k_test[0]])
        cnn.save(cnn_path)


def load_preprocessed_data(dataset: str):
    print("!!!", local_dir)
    return np.load(local_dir.joinpath(f'pre_processed_data/{dataset}/{dataset}_preprocessed_data_{ACC_HZ}.npy'),
                   allow_pickle=True).item()

def load_and_train(max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1, use_logits = False, n_classes=4):

    static_preprocessed_data = load_preprocessed_data("stationary")
    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data, 
                                      n_classes=n_classes)

    # hybrid_preprocessed_data = load_preprocessed_data("hybrid")
    # hybrid_data_bundle = prepare_data(hybrid_preprocessed_data, 
    #                                   n_classes=n_classes)

    start_time = time.time()

    # Define the learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Metric to monitor
        factor=0.5,          # Factor by which the learning rate will be reduced
        patience=16,         # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-5          # Lower bound on the learning rate
    )


    train_rgb_cnn(
        static_keys,
        static_data_bundle,
        None, # hybrid_data_bundle,
        fit_callbacks=[reduce_lr],
        max_splits=max_splits,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_logits=use_logits,
        n_classes=n_classes
    )
    # train_logreg(static_keys, static_data_bundle)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # do_preprocessing(big_specgram_process, cache_dir=local_dir.joinpath("pre_processed_data"))
    load_and_train(epochs=2, batch_size=1, lr=1e-4, use_logits=False, n_classes=2)

# 0. focus on sleep wake
# 1. trim input size down (either downsample img in-network or pre-network)
# 2. increase number of easy wake epochs
# 3. explore more input transformations: 
#   i. xyz amplitude before specgram (vs learning RGB + rotations)
#   ii. different specgram types (different parameters)
