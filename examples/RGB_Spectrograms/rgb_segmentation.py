import json
import os

# Suppress TF warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#[Compiling module a_inference_one_step_on_data_58306__.69169] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
# os.environ['XLA_FLAGS'] = '--xla_dump_to=./xla_dump'

# Use jax backend
# on macOS, this is one of the better out-of-the-box GPU options
# we have to do this first, before importing Keras ANYWHERE (including in pisces/other modules)
# So ignore the warnings about imports below this line
# pylint: disable=wrong-import-position,wrong-import-order
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# from examples.RGB_Spectrograms.losses import MaskedTemporalCategoricalCrossEntropy
# from examples.NHRC.src.preprocess_and_save import preprocessed_data_filename
from examples.RGB_Spectrograms.plotting import create_histogram_rgb, debug_plot
import sys
from pathlib import Path
from typing import List

from examples.RGB_Spectrograms.utils import load_preprocessed_data

local_dir = Path(__file__).resolve().parent
print("local_dir: ", local_dir)
sys.path.append(str(local_dir.parent.joinpath('NHRC')))
from examples.RGB_Spectrograms.preprocessing import PreparedDataRGB, big_specgram_process, prepare_data, do_preprocessing
# from examples.RGB_Spectrograms.channel_permuter import PermutationDataGenerator, Random3DRotationGenerator
from examples.RGB_Spectrograms.models import segmentation_model
from examples.RGB_Spectrograms.constants import ACC_HZ, NEW_INPUT_SHAPE, N_OUTPUT_EPOCHS, PSG_MASK_VALUE, rgb_path_name, rgb_saved_predictions_name
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
from keras.metrics import SpecificityAtSensitivity


from pisces.metrics import WASAMetric, wasa_metric

# downsample rate for frequency axis
# FREQ_DOWN = 4 #NEW_INPUT_SHAPE[1] // 16 
FREQ_DOWN = NEW_INPUT_SHAPE[1] // 16 

def rgb_gather_reshape(data_bundle: PreparedDataRGB, idx_tensor: np.array, input_shape: tuple, output_shape: tuple) -> tuple | None:
    input_shape_stack = (-1, *input_shape)
    output_shape_stack = (-1, *output_shape)

    idx_data = data_bundle.spectrograms[idx_tensor].reshape(input_shape_stack)
    idx_labels = data_bundle.labels[idx_tensor].reshape(output_shape_stack)
    idx_sample_weights = data_bundle.weights[idx_tensor].reshape(output_shape_stack)

    idx_specgram_mean = channelwise_mean(idx_data)
    idx_specgram_std = channelwise_std(idx_data)

    idx_data_centered = (idx_data - idx_specgram_mean) / idx_specgram_std
    
    return idx_data_centered, idx_labels, idx_sample_weights


def channelwise_mean(x, axes=[1, 2]):
    for axis in axes:
        x = np.mean(x, axis=axis, keepdims=True)
    return x

def channelwise_std(x, axes=[1, 2]):
    for axis in axes:
        x = np.std(x, axis=axis, keepdims=True)
    return x

def train_rgb_cnn(static_keys, static_data_bundle, hybrid_data_bundle, fit_callbacks: list = [], max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1, use_logits = False, n_classes=4, predictions_path: Path = None, models_path: Path = None):
    split_maker = LeaveOneOut()

    training_results = []
    cnn_predictors = []

    print(f"Training RGB CNN models with {n_classes} classes...")
    WASA_PERCENT = 95
    WASA_FRAC = WASA_PERCENT / 100
    # os.makedirs("./saved_outputs", exist_ok=True)

    INPUT_SHAPE = list(NEW_INPUT_SHAPE)
    INPUT_SHAPE[1] //= FREQ_DOWN
    INPUT_SHAPE = tuple(INPUT_SHAPE)
    print("INPUT SHAPE: ", INPUT_SHAPE)

    OUTPUT_SHAPE = (N_OUTPUT_EPOCHS, )


    def make_segmenter():
        return segmentation_model(input_shape=INPUT_SHAPE, num_classes=1, from_logits=use_logits)
    
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
        test_id = static_keys[k_test[0]]

        # network instance to be trained
        cnn = make_segmenter()

        # Get the data
        train_data, train_labels, train_sample_weights = rgb_gather_reshape(
            static_data_bundle, train_idx_tensor,
            input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)

        test_data, test_labels, test_sample_weights = rgb_gather_reshape(
            static_data_bundle, test_idx_tensor,
            input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)

        # Train the model on the training set
        # output = cnn.layers[-1].output
        # print("output shape: ", output.shape)

        # print("INPUT SHAPE: ", INPUT_SHAPE)
        # print("train input shape: ", train_data.shape)
        # print("train label shape: ", train_labels.shape)
        # print("test label shape: ", test_labels.shape)
        # print("unique train labels: ", np.unique(train_labels))

        # any mods to the labels
        # train_labels_pro = tf.zeros_like(train_labels)
        train_labels_pro = tf.where(
            train_labels == PSG_MASK_VALUE, 0, train_labels)

        # test_labels_pro = tf.zeros_like(test_labels)
        test_labels_pro = tf.where(
            test_labels == PSG_MASK_VALUE, 0, test_labels)

        # temporal_ce = MaskedTemporalCategoricalCrossEntropy(n_classes=n_classes, sparse=True, from_logits=use_logits)

        cnn.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=use_logits),
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            weighted_metrics=[
                # SpecificityAtSensitivity(WASA_FRAC, name=f"WASA{WASA_PERCENT}"),
                # 'auc'
            ]
        )

        # channel_shuffler = PermutationDataGenerator(train_data, train_labels, sample_weights=train_sample_weights, batch_size=batch_size) 

        print("Training model...")

        training_results.append(cnn.fit(
            train_data, train_labels_pro,
            # channel_shuffler,
            sample_weight=train_sample_weights,
            epochs=epochs,
            # validation_data=(test_data, test_labels),
            validation_data=(test_data, test_labels_pro, test_sample_weights),
            batch_size=batch_size,
            callbacks=[cnn_tensorboard_callback, *fit_callbacks]
        ))

        cnn_predictors.append(cnn)

        evaluate_and_save_test(
            test_data,
            test_labels_pro,
            test_sample_weights,
            cnn,
            test_id,
            "static",
            use_logits,
            wasa_percent=WASA_PERCENT,
            predictions_path=predictions_path
        )

        # Now do the same with the hybrid data
        hybrid_test_data, _, hybrid_test_sample_weights = rgb_gather_reshape(
            hybrid_data_bundle, test_idx_tensor,
            input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
        
        evaluate_and_save_test(
            hybrid_test_data,
            test_labels_pro, # labels SHOULD be the same
            hybrid_test_sample_weights,
            cnn,
            test_id,
            "hybrid",
            use_logits,
            wasa_percent=WASA_PERCENT,
            predictions_path=predictions_path
        )


        # Use cnn to predict probabilities
        # Rescale so everything doesn't get set to 0 or 1 in the expit call
        # scalar = 10000.0 if use_logits else 1.0 # DONT USE HERE....pushed everything to 0.5, i.e. expit(0)

        # save the trained model weights
        if models_path is not None:
            cnn_path = rgb_path_name(test_id, saved_model_dir=models_path)
            cnn.save(cnn_path)

def evaluate_and_save_test(
        test_data,
        test_labels,
        test_sample_weights,
        cnn,
        test_id: str,
        set_name: str,
        use_logits: bool,
        wasa_percent: int = 95,
        predictions_path: Path = None):
    test_labels = test_labels[0]
    test_sample_weights = test_sample_weights[0]
    test_prediction_raw = cnn.predict(test_data)
    test_probabilities = expit(test_prediction_raw) if use_logits else test_prediction_raw
    print("Plotting predictions")
    wasa_computed = True
    try:
        # p_sleep = 1 - test_probabilities[:, 0]
        p_wake = 1 - test_probabilities
        wasa, threshold = wasa_metric(
            labels=np.squeeze(test_labels),
            predictions=np.squeeze(p_wake),
            weights=np.squeeze(test_sample_weights))
        print(f"WASA{wasa_percent}: {wasa.wake_accuracy:.4f}")
        test_pred_path = Path(rgb_saved_predictions_name(test_id, saved_output_dir=predictions_path, set_name=set_name))
        print(f"Saving predictions to {test_pred_path}")
        # test_pred = (1 - p_wake)> threshold
        np.save(test_pred_path, test_probabilities)
    except Exception as e:
        print(f"Error computing WASA: {e}")
        wasa_computed = False
    debug_plot(
        predictions=test_probabilities,
        spectrogram_3d=test_data, 
        y_true=test_labels,
        weights=test_sample_weights,
        saveto=predictions_path.joinpath(f"{test_id}_cnn_pred_{set_name}_{ACC_HZ}.png")
                                         if predictions_path is not None else None)
    


def load_and_train(preprocessed_path: Path, max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1, use_logits = False, n_classes=4, predictions_path: str = None):

    static_preprocessed_data = load_preprocessed_data("stationary", preprocessed_path)
    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data, 
                                      n_classes=n_classes,
                                      freq_downsample=FREQ_DOWN)

    hybrid_preprocessed_data = load_preprocessed_data("hybrid", preprocessed_path)
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data, 
                                      n_classes=n_classes,
                                      freq_downsample=FREQ_DOWN)

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
        hybrid_data_bundle,
        fit_callbacks=[reduce_lr],
        max_splits=max_splits,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_logits=use_logits,
        n_classes=n_classes,
        predictions_path=predictions_path
    )
    # train_logreg(static_keys, static_data_bundle)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    preprocessed_data_path = local_dir.joinpath("pre_processed_data")
    predictions_path = local_dir.joinpath("saved_outputs")

    # do_preprocessing(big_specgram_process, cache_dir=preprocessed_data_path)
    load_and_train(preprocessed_path=preprocessed_data_path, epochs=2, batch_size=1, lr=1e-4, use_logits=True, n_classes=2, predictions_path=predictions_path)
    create_histogram_rgb(
        "rgb", 
        preprocessed_data_path=preprocessed_data_path,  
        saved_output_dir=predictions_path, 
        training_prediction_path=predictions_path)


# 0. focus on sleep wake
# 1. trim input size down (either downsample img in-network or pre-network)
# 2. increase number of easy wake epochs
# 3. explore more input transformations: 
#   i. xyz amplitude before specgram (vs learning RGB + rotations)
#   ii. different specgram types (different parameters)
