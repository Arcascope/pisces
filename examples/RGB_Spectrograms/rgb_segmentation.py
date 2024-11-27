import os
import sys
from pathlib import Path
from typing import List


# Use jax backend
# on macOS, this is one of the better out-of-the-box GPU options
# we have to do this first, before importing Keras ANYWHERE (including in pisces/other modules)
# So ignore the warnings about imports below this line
# pylint: disable=wrong-import-position,wrong-import-order
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

local_dir = Path(__file__).resolve().parent
print("local_dir: ", local_dir)
sys.path.append(str(local_dir.parent.joinpath('NHRC')))
from examples.RGB_Spectrograms.preprocessing import big_specgram_process
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


def overlay_channels_fixed(spectrogram_tensor, mintile=5, maxtile=95, ax=None):
    """
    Overlay spectrogram channels as an RGB image by stacking the three axes (x, y, z).
    
    Parameters:
        spectrogram_tensor (numpy.ndarray): Spectrogram tensor of shape (time_bins, freq_bins, 3).
    """
    # Normalize each channel to [0, 1] for proper RGB visualization
    norm_spec = np.zeros_like(spectrogram_tensor)
    for i in range(3):
        channel = spectrogram_tensor[:, :, i]
        p5, p95 = np.percentile(channel, [mintile, maxtile])  # Robust range
        norm_spec[:, :, i] = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)  # Avoid dividing by zero
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Display the combined RGB image

    ax.imshow(norm_spec, aspect='auto', origin='lower')
    # add_rgb_legend(plt.gca())
    # plt.colorbar(label='Intensity')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Frequency Bins')
    ax.set_title('Overlayed Spectrogram Channels as RGB')


def compute_stage_weights(label_stack):
    # Compute balancing sample weights based on label_stack
    # equal weights 
    sample_weights =0.25 + np.zeros_like(label_stack, dtype=np.float32)
    sample_weights[label_stack < 0] = 0.0
    return sample_weights

    # for idx in range(len(label_stack)):
    #     substack = label_stack[idx]
    #     n_scored = np.sum(substack >= 0)
    #     for i in range(4):
    #         sample_weights[idx][substack == i] = n_scored / (np.sum(substack == i) + 1e-8)

    #     sample_weights[idx] /= np.sum(sample_weights[idx])
    
    # return sample_weights


@dataclass
class PreparedDataRGB:
    spectrograms: np.array
    labels: np.array 
    weights: np.array

def rgb_gather_reshape(data_bundle: PreparedDataRGB, train_idx_tensor: np.array, input_shape: tuple = NEW_INPUT_SHAPE, output_shape: tuple = (N_OUTPUT_EPOCHS,)) -> tuple | None:
    input_shape = (-1, *input_shape)
    output_shape = (-1, *output_shape)
    train_data = data_bundle.spectrograms[train_idx_tensor].reshape(input_shape)
    train_labels = data_bundle.labels[train_idx_tensor].reshape(output_shape)
    train_sample_weights = data_bundle.weights[train_idx_tensor].reshape(output_shape)
    
    return train_data, train_labels, train_sample_weights

def prepare_data(preprocessed_data) -> PreparedDataRGB:
    label_stack = np.array([
        stages_map(preprocessed_data[k]['psg'][:, 1])
        for k in list(preprocessed_data.keys())
    ])

    label_weights = compute_stage_weights(label_stack)

    label_weights[label_stack < 0] = 0.0

    label_stack_masked = np.zeros_like(label_stack)
    label_stack_masked[label_stack >= 0] = label_stack[label_stack >= 0]

    spectrogram_stack = np.array([
        preprocessed_data[k]['spectrogram']
        for k in list(preprocessed_data.keys())
    ])

    # clip the spectrogram stack to 0.05 and 0.95 quantiles
    p_low = 10
    for i in range(3):
        p5, p95 = np.percentile(spectrogram_stack[:, :, :, i], [p_low, 100 - p_low])
        spectrogram_stack[:, :, :, i] = np.clip(spectrogram_stack[:, :, :, i], p5, p95)

    return PreparedDataRGB(
        spectrograms=spectrogram_stack.astype(np.float32),
        labels=label_stack_masked.astype(np.float32),
        weights=label_weights.astype(np.float32)
    )

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

def train_rgb_cnn(static_keys, static_data_bundle, hybrid_data_bundle, fit_callbacks: list = [], max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1):
    


    split_maker = LeaveOneOut()

    training_results = []
    cnn_predictors = []

    print(f"Training RGB CNN models...")
    WASA_PERCENT = 95
    WASA_FRAC = WASA_PERCENT / 100

    # Split the data into training and testing sets
    for k_train, k_test in tqdm(split_maker.split(static_keys), desc="Next split", total=len(static_keys)):
        log_dir_cnn = f"./logs/rgb_cnn_{static_keys[k_test[0]]}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # Configure TensorBoard callback
        cnn_tensorboard_callback = TensorBoard(
            log_dir=log_dir_cnn, histogram_freq=1)
        if (max_splits > 0) and (len(training_results) >= max_splits):
                break
        # Convert indices to tensors
        # train_idx_tensor = tf.constant(k_train, dtype=tf.int32)
        # test_idx_tensor = tf.constant(k_test, dtype=tf.int32)
        train_idx_tensor = np.array(k_train)
        test_idx_tensor = np.array(k_test)

        # training
        train_data, train_labels, train_sample_weights = rgb_gather_reshape(
            static_data_bundle, train_idx_tensor)
        # Evaluate the model on the test data
        test_data, test_labels, test_sample_weights = rgb_gather_reshape(
            static_data_bundle, test_idx_tensor)

        # Train the model on the training set
        from_logits = False 
        cnn = segmentation_model(from_logits=from_logits)

        cnn.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
            optimizer=keras.optimizers.AdamW(learning_rate=lr),
            weighted_metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                WASAMetric(sleep_accuracy=WASA_FRAC, from_logits=from_logits)
            ]
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

        channel_shuffler = PermutationDataGenerator(train_data, train_labels, sample_weights=train_sample_weights, batch_size=batch_size) 
        # channel_shuffler = Random3DRotationGenerator(train_data, train_labels, train_sample_weights, batch_size=batch_size)


        training_results.append(cnn.fit(
            channel_shuffler,
            # train_data, train_labels,
            # sample_weight=train_sample_weights,
            epochs=epochs,
            validation_data=(test_data, test_labels, test_sample_weights),
            # batch_size=batch_size, # 4 seems to be the max we can handle for cnn.predict(stack_of_spectrograms) on M3 Max w/ 64 gb of RAM
            callbacks=[cnn_tensorboard_callback, *fit_callbacks]
        ))

        cnn_predictors.append(cnn)


        # Use cnn to predict probabilities
        # Rescale so everything doesn't get set to 0 or 1 in the expit call
        # scalar = 10000.0
        # test_prediction_raw = test_prediction_raw / scalar
        # test_pred = expit(test_prediction_raw).reshape(-1,)
        test_prediction_raw = cnn.predict(test_data)[0]
        test_data = test_data[0]
        test_labels = test_labels[0]
        test_sample_weights = test_sample_weights[0]
        if from_logits:
            print("Applying softmax")
            test_prediction_raw = softmax(test_prediction_raw, axis=-1)
        print("Plotting predictions")
        os.makedirs("./saved_outputs", exist_ok=True)
        debug_plot(
            test_prediction_raw, 
            test_data, 
            weights=test_sample_weights,
            saveto=f"./saved_outputs/{static_keys[k_test[0]]}_cnn_pred_static_{ACC_HZ}.png")
        test_pred = test_prediction_raw

        wasa = wasa_metric(
            labels=test_labels,
            predictions=np.sum(test_pred[:, 1:], axis=-1),
            weights=test_sample_weights)
        print(f"WASA{WASA_PERCENT}: {wasa.wake_accuracy:.4f}")
        # test_pred_path = (static_keys[k_test[0]]) + \
        #     f"_cnn_pred_static_{ACC_HZ}.npy"
        # np.save("./saved_outputs/" + test_pred_path, test_pred)

        # Repeat for hybrid data
        # Evaluate the model on the test data
        # test_data, test_labels, test_sample_weights = rgb_gather_reshape(
        #     hybrid_data_bundle, test_idx_tensor)

        # Use cnn to predict probabilities
        # test_prediction_raw = cnn.predict(test_data)
        # test_prediction_raw = test_prediction_raw / scalar
        # test_pred = expit(test_prediction_raw).reshape(-1,)
        # test_pred = test_prediction_raw
        # test_pred_path = (static_keys[k_test[0]]) + \
        #     f"_cnn_pred_hybrid_{ACC_HZ}.npy"
        # np.save("saved_outputs/" + test_pred_path, test_pred)

        # save the trained model weights
        cnn_path = rgb_path_name(static_keys[k_test[0]])
        cnn.save(cnn_path)

def debug_plot(predictions, spectrogram_3d, weights: np.ndarray | None = None, saveto: str = None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    overlay_channels_fixed(np.swapaxes(spectrogram_3d, 0, 1), ax=axs[0])
    axs[1].stackplot(range(N_OUTPUT_EPOCHS), predictions.T)
    axs[1].set_xlim([0, N_OUTPUT_EPOCHS])
    axs[1].set_ylim([0, 1])
    if weights is not None:
        # apply gray vertical bar over any regions with weight 0.0
        for idx in np.where(weights == 0.0)[0]:
            axs[1].axvspan(idx, idx+1, color='gray', alpha=0.5)
    fig.tight_layout(pad=0.1)
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto)
    plt.close()

def load_preprocessed_data(dataset: str):
    print("!!!", local_dir)
    return np.load(local_dir.joinpath(f'pre_processed_data/{dataset}/{dataset}_preprocessed_data_{ACC_HZ}.npy'),
                   allow_pickle=True).item()

def load_and_train(max_splits: int = -1, epochs: int = 1, lr: float = 1e-4, batch_size: int = 1):

    static_preprocessed_data = load_preprocessed_data("stationary")
    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data)

    hybrid_preprocessed_data = load_preprocessed_data("hybrid")
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    start_time = time.time()

    # Define the learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Metric to monitor
        factor=0.5,          # Factor by which the learning rate will be reduced
        patience=max(1, epochs // 8),          # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6          # Lower bound on the learning rate
    )


    train_rgb_cnn(
        static_keys,
        static_data_bundle,
        hybrid_data_bundle,
        fit_callbacks=[reduce_lr],
        max_splits=max_splits,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    # train_logreg(static_keys, static_data_bundle)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # do_preprocessing(big_specgram_process)
    load_and_train(epochs=10, batch_size=1, lr=1e-3)
