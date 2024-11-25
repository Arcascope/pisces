import os

# Use jax backend
# on macOS, this is one of the better out-of-the-box GPU options
# we have to do this first, before importing Keras ANYWHERE (including in pisces/other modules)
# So ignore the warnings about imports below this line
# pylint: disable=wrong-import-position,wrong-import-order
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dataclasses import dataclass
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation,
    AveragePooling2D, Reshape, GlobalAveragePooling2D, Lambda
)
from keras.models import Model


from src.constants import ACC_HZ
from examples.NHRC.wasa_metric import BinaryTruePositives, SleepAccuracyMetric
from src.preprocess_and_save import do_preprocessing, big_specgram_process
from nhrc_utils.analysis import stages_map





def add_rgb_legend(ax):
    """
    Adds an RGB legend indicating the mapping of colors to accelerometer axes.
    """
    
    # Create a small color block with labels
    colors = ['red', 'green', 'blue']
    labels = ['X-axis (Red)', 'Y-axis (Green)', 'Z-axis (Blue)']
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.2, i + 0.5, labels[i], va='center', fontsize=12)

    ax.axis('off')
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 3)

def overlay_channels_fixed(spectrogram_tensor, mintile=5, maxtile=95):
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
    
    # Display the combined RGB image
    plt.figure(figsize=(10, 5))
    plt.imshow(norm_spec, aspect='auto', origin='lower')
    # add_rgb_legend(plt.gca())
    # plt.colorbar(label='Intensity')
    plt.xlabel('Time Bins')
    plt.ylabel('Frequency Bins')
    plt.title('Overlayed Spectrogram Channels as RGB')
    # plt.show()


# from examples.NHRC.nhrc_utils.new_cnn import NEW_INPUT_SHAPE
NEW_INPUT_SHAPE = (15360, 257, 3)
NEW_OUTPUT_SHAPE = (1024, 4)

# %%
def debug_normalization(spectrogram_tensor):
    for i in range(3):
        channel = spectrogram_tensor[:, :, i]
        print(f"Channel {i} - Min: {channel.min()}, Max: {channel.max()}, Mean: {channel.mean()}")



def segmentation_model(input_shape=NEW_INPUT_SHAPE, num_classes=4):
    inputs = Input(shape=input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # Downsampling
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # Further Downsampling
    
    # c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    # c3 = BatchNormalization()(c3)
    # p3 = MaxPooling2D((2, 2))(c3)  # Bottleneck
    
    # # Decoder
    # u1 = UpSampling2D((2, 2))(p3)
    # u1 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    # u1 = BatchNormalization()(u1)
    # u1 = Concatenate()([u1, c3])  # Skip connection
    
    u2 = UpSampling2D((2, 2))(p2)
    u2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Concatenate()([u2, c2])  # Skip connection
    u2 = UpSampling2D((2, 2))(u2)
    
    # Collapse frequency axis
    collapse = AveragePooling2D(pool_size=(1, u2.shape[2]))(u2)  # Collapse frequency (128 -> 1)

    # Downsample to match label shape
    final_downsampling = AveragePooling2D(pool_size=(15, 1))(collapse)  # Downsample time (* -> 1024)
    
    # Final dense layer for class probabilities
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(final_downsampling)  # (1024, 4, 1)
    # outputs = Lambda(lambda x: tf.squeeze(x, axis=2))(outputs)  # Remove leftover spatial dimensions (1024, 4)
    # outputs = Lambda(lambda x: jnp.squeeze(x, axis=2))(outputs)  # Remove leftover spatial dimensions (1024, 4)
    outputs = Reshape(NEW_OUTPUT_SHAPE)(outputs)
    
    model = Model(inputs, outputs)
    return model

# %%
def compute_stage_weights(label_stack):
    stage_counts = np.zeros(4)
    for i in range(4):
        stage_counts[i] = np.sum(label_stack == i)
    
    stage_weights = 1 / stage_counts
    return stage_weights

# %% [markdown]
# # LOOX training loop

# %%


# Define the learning rate scheduler callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Metric to monitor
    factor=0.5,          # Factor by which the learning rate will be reduced
    patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6          # Lower bound on the learning rate
)

@dataclass
class PreparedDataRGB:
    spectrograms: np.array
    labels: np.array 
    weights: np.array

OUTPUT_SHAPE = (1024,)

def rgb_gather_reshape(data_bundle: PreparedDataRGB, train_idx_tensor: np.array, input_shape: tuple = NEW_INPUT_SHAPE, output_shape: tuple = OUTPUT_SHAPE) -> tuple | None:
    input_shape = (-1, *input_shape)
    output_shape = (-1, *output_shape)
    train_data = data_bundle.spectrograms[train_idx_tensor].reshape(input_shape)
    train_labels = data_bundle.labels[train_idx_tensor].reshape(output_shape)
    train_sample_weights = data_bundle.weights[train_idx_tensor].reshape(output_shape)
    
    return train_data, train_labels, train_sample_weights

def prepare_data(preprocessed_data):
    label_stack = np.array([
        stages_map(preprocessed_data[k]['psg'][:, 1])
        for k in list(preprocessed_data.keys())
    ])

    label_weights = np.zeros_like(label_stack, dtype=np.float32)
    stage_weights = compute_stage_weights(label_stack)
    for i in range(4):
        label_weights[label_stack == i] = stage_weights[i]

    label_weights[label_stack < 0] = 0.0

    label_stack_masked = np.zeros_like(label_stack)
    label_stack_masked[label_stack >= 0] = label_stack[label_stack >= 0]

    spectrogram_stack = np.array([
        preprocessed_data[k]['spectrogram']
        for k in list(preprocessed_data.keys())
    ])

    return PreparedDataRGB(
        spectrograms=spectrogram_stack.astype(np.float32),
        labels=label_stack_masked.astype(np.float32),
        weights=label_weights.astype(np.float32)
    )

def rgb_path_name(key):
    os.makedirs("./saved_models", exist_ok=True)
    return f"./saved_models/rgb_{key}_{ACC_HZ}.keras"

def train_rgb_cnn(static_keys, static_data_bundle, hybrid_data_bundle, max_splits: int = -1, epochs: int = 1, lr: float = 1e-4):
    
    log_dir_cnn = f"./logs/rgb_cnn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Configure TensorBoard callback
    cnn_tensorboard_callback = TensorBoard(
        log_dir=log_dir_cnn, histogram_freq=1)

    split_maker = LeaveOneOut()

    training_results = []
    cnn_predictors = []

    print(f"Training RGB CNN models...")
    WASA_PERCENT = 95
    WASA_FRAC = WASA_PERCENT / 100

    # Split the data into training and testing sets
    for k_train, k_test in tqdm(split_maker.split(static_keys), desc="Next split", total=len(static_keys)):
        if (max_splits > 0) and (len(training_results) >= max_splits):
                break
        # Convert indices to tensors
        train_idx_tensor = tf.constant(k_train, dtype=tf.int32)
        test_idx_tensor = tf.constant(k_test, dtype=tf.int32)

        # training
        train_data, train_labels, train_sample_weights = rgb_gather_reshape(
            static_data_bundle, train_idx_tensor)
        # Evaluate the model on the test data
        test_data, test_labels, test_sample_weights = rgb_gather_reshape(
            static_data_bundle, test_idx_tensor)
        print("Train data created")

        # Train the model on the training set
        cnn = segmentation_model()
        print("segmentation model created")

        cnn.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.AdamW(learning_rate=lr),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                # 'accuracy',
                # BinaryTruePositives(),
                # keras.metrics.SensitivityAtSpecificity(
                #     WASA_FRAC,
                #     num_thresholds=200,
                #     class_id=0,
                #     name=f'WASA{WASA_PERCENT}',
                #     dtype=None)
                ]
        )

        
        # gives weight 0 to -1 "mask" intervals, 1 to the rest

        # make the labels binary, -1 -> 0
        # since we incorporate the mask in the sample weights,
        # we can just set the labels to 0

        training_results.append(cnn.fit(
            train_data, train_labels,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            batch_size=1, # 4 seems to be the max we can handle for cnn.predict(stack_of_spectrograms) on M3 Max w/ 64 gb of RAM
            sample_weight=train_sample_weights,
            # callbacks=[cnn_tensorboard_callback, reduce_lr]
            callbacks=[reduce_lr]
        ))

        cnn_predictors.append(cnn)


        # Use cnn to predict probabilities
        # Rescale so everything doesn't get set to 0 or 1 in the expit call
        # scalar = 10000.0
        test_prediction_raw = cnn.predict(test_data)
        # test_prediction_raw = test_prediction_raw / scalar
        # test_pred = expit(test_prediction_raw).reshape(-1,)
        test_pred = test_prediction_raw
        test_pred_path = (static_keys[k_test[0]]) + \
            f"_cnn_pred_static_{ACC_HZ}.npy"
        os.makedirs("./saved_outputs", exist_ok=True)
        np.save("./saved_outputs/" + test_pred_path, test_pred)

        # Repeat for hybrid data
        # Evaluate the model on the test data
        test_data, test_labels, test_sample_weights = rgb_gather_reshape(
            hybrid_data_bundle, test_idx_tensor)

        # Use cnn to predict probabilities
        test_prediction_raw = cnn.predict(test_data)
        # test_prediction_raw = test_prediction_raw / scalar
        # test_pred = expit(test_prediction_raw).reshape(-1,)
        test_pred = test_prediction_raw
        test_pred_path = (static_keys[k_test[0]]) + \
            f"_cnn_pred_hybrid_{ACC_HZ}.npy"
        np.save("saved_outputs/" + test_pred_path, test_pred)

        # save the trained model weights
        cnn_path = rgb_path_name(static_keys[k_test[0]])
        cnn.save(cnn_path)

def load_preprocessed_data(dataset: str):
    return np.load(f'./pre_processed_data/{dataset}/{dataset}_preprocessed_data_{ACC_HZ}.npy',
                   allow_pickle=True).item()

def load_and_train(max_splits: int = -1, epochs: int = 1, lr: float = 1e-4):

    static_preprocessed_data = load_preprocessed_data("stationary")
    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data)

    hybrid_preprocessed_data = load_preprocessed_data("hybrid")
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    start_time = time.time()

    train_rgb_cnn(
        static_keys,
        static_data_bundle,
        hybrid_data_bundle,
        max_splits=max_splits,
        epochs=epochs,
        lr=lr
    )
    # train_logreg(static_keys, static_data_bundle)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

# %%
if __name__ == "__main__":
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # do_preprocessing(big_specgram_process)
    load_and_train(epochs=1, lr=1e-3)

# %%



