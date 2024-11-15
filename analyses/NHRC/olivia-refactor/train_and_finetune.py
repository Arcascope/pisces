
from analyses.NHRC.nhrc_utils.model_definitions import FINETUNING_INPUT_SHAPE, build_finetuning_model, EXTRA_LAYERS_NAME
from analyses.NHRC.nhrc_utils.model_definitions import LR_CNN_NAME,  LABEL_SHAPE, LR_INPUT_SHAPE, WeightedModel, build_lr_cnn
import numpy as np
import tensorflow as tf
import pisces.models as pm
import datetime
from analyses.NHRC.nhrc_utils.analysis import make_lr_filename, make_finetuning_filename
from analyses.NHRC.nhrc_utils.model_definitions import LR_INPUT_LENGTH
from analyses.NHRC.nhrc_utils.analysis import prepare_data
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
from sklearn.calibration import expit
import time
from constants import ACC_HZ as acc_hz
import os
import keras


# This is producing weird results for me
def train_logreg(static_keys, static_data_bundle, hybrid_data_bundle):

    # Set up separate log directories for each model
    log_dir_lr = f"./logs/lr_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    lr_tensorboard_callback = TensorBoard(log_dir=log_dir_lr, histogram_freq=1)

    split_maker = pm.LeaveOneOutSplitter()

    training_results = []
    lr_predictors = []

    print(f"Training {LR_CNN_NAME} models...")

    # Split the data into training and testing sets
    for k_train, k_test in tqdm(split_maker.split(static_keys), desc="Next split", total=len(static_keys)):
        # Convert indices to tensors
        train_idx_tensor = tf.constant(k_train, dtype=tf.int32)
        test_idx_tensor = tf.constant(k_test, dtype=tf.int32)

        # Gather the training and validation data using tf.gather
        # training
        train_data = tf.reshape(
            tf.gather(static_data_bundle.activity, train_idx_tensor),
            LR_INPUT_SHAPE)
        train_labels = tf.reshape(
            tf.gather(static_data_bundle.true_labels, train_idx_tensor),
            LABEL_SHAPE)
        train_sample_weights = tf.reshape(
            tf.gather(static_data_bundle.sample_weights, train_idx_tensor),
            LABEL_SHAPE)

        # make the labels binary, -1 -> 0
        # since we incorporate the mask in the sample weights, we can just set the labels to 0
        train_labels_masked = tf.where(
            train_sample_weights > 0, train_labels, 0.0)

        # z-normalize input data
        train_data = (train_data - tf.reduce_mean(train_data)) / \
            np.std(train_data)

        # Custom loss function that includes the sample weights
        lr_cnn = build_lr_cnn()
        weighted_lr_cnn = WeightedModel(lr_cnn)
        weighted_lr_cnn.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (train_data, train_labels_masked, train_sample_weights))
        dataset = dataset.batch(32)
        training_results.append(weighted_lr_cnn.fit(
            dataset,
            epochs=350,
            verbose=0,
            callbacks=[lr_tensorboard_callback]
        ))

        # Evaluate the trained model on test_idx_tensor
        test_data = tf.reshape(
            tf.gather(static_data_bundle.activity, test_idx_tensor),
            LR_INPUT_SHAPE)

        # z-normalize input data
        test_data = (test_data - tf.reduce_mean(test_data)) / \
            np.std(test_data)

        # Evaluate
        test_pred = expit(
            lr_cnn.predict(
                tf.reshape(test_data, (1, LR_INPUT_LENGTH, 1)),
                verbose=0
            )).reshape(-1,)

        # Save test_pred to a file
        test_pred_path = (static_keys[k_test[0]]) + "_logreg_pred.npy"
        np.save(test_pred_path, test_pred)

        lr_predictors.append(lr_cnn)
        lr_path = make_lr_filename(static_keys[k_test[0]])
        lr_cnn.save(lr_path)


def finetuning_gather_reshape(data_bundle, train_idx_tensor: tf.Tensor, input_shape: tuple = FINETUNING_INPUT_SHAPE, output_shape: tuple = LABEL_SHAPE) -> tuple | None:
    train_data = tf.reshape(
        tf.gather(data_bundle.mo_predictions, train_idx_tensor),
        input_shape
    )
    train_labels = tf.reshape(
        tf.gather(data_bundle.true_labels, train_idx_tensor),
        output_shape)
    train_sample_weights = tf.reshape(
        tf.gather(data_bundle.sample_weights, train_idx_tensor),
        output_shape)
    return train_data, train_labels, train_sample_weights


def train_cnn(static_keys, static_data_bundle, hybrid_data_bundle):

    log_dir_cnn = f"./logs/cnn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Configure TensorBoard callback
    cnn_tensorboard_callback = TensorBoard(
        log_dir=log_dir_cnn, histogram_freq=1)

    split_maker = pm.LeaveOneOutSplitter()

    training_results = []
    cnn_predictors = []

    print(f"Training {EXTRA_LAYERS_NAME} models...")

    # Split the data into training and testing sets
    for k_train, k_test in tqdm(split_maker.split(static_keys), desc="Next split", total=len(static_keys)):
        # Convert indices to tensors
        train_idx_tensor = tf.constant(k_train, dtype=tf.int32)
        test_idx_tensor = tf.constant(k_test, dtype=tf.int32)

        # Gather the training and validation data using tf.gather
        # training
        train_data, train_labels, train_sample_weights = finetuning_gather_reshape(
            static_data_bundle, train_idx_tensor)

        # make the labels binary, -1 -> 0
        # since we incorporate the mask in the sample weights, we can just set the labels to 0
        train_labels_masked = tf.where(
            train_sample_weights > 0, train_labels, 0.0)

        # Train the model on the training set
        cnn = build_finetuning_model(FINETUNING_INPUT_SHAPE[1:])

        cnn.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=keras.optimizers.AdamW(learning_rate=5e-4),
        )

        # gives weight 0 to -1 "mask" intervals, 1 to the rest

        # make the labels binary, -1 -> 0
        # since we incorporate the mask in the sample weights,
        # we can just set the labels to 0
        train_labels_masked = np.where(train_sample_weights, train_labels, 0)

        training_results.append(cnn.fit(
            train_data, train_labels_masked,
            epochs=100,
            validation_split=0.0,
            batch_size=1,
            sample_weight=train_sample_weights,
            verbose=0,
            callbacks=[cnn_tensorboard_callback]
        ))

        cnn_predictors.append(cnn)
        cnn_path = make_finetuning_filename(static_keys[k_test[0]])

        # Evaluate the model on the test data
        test_data, test_labels, test_sample_weights = finetuning_gather_reshape(
            static_data_bundle, test_idx_tensor)

        # Use cnn to predict probabilities
        # Rescale so everything doesn't get set to 0 or 1 in the expit call
        scalar = 10000.0
        test_prediction_raw = cnn.predict(test_data)
        test_prediction_raw = test_prediction_raw / scalar
        test_pred = expit(test_prediction_raw).reshape(-1,)
        test_pred_path = (static_keys[k_test[0]]) + \
            f"_cnn_pred_static_{acc_hz}.npy"
        os.makedirs("./saved_outputs", exist_ok=True)
        np.save("./saved_outputs/" + test_pred_path, test_pred)

        # Repeat for hybrid data
        # Evaluate the model on the test data
        test_data, test_labels, test_sample_weights = finetuning_gather_reshape(
            hybrid_data_bundle, test_idx_tensor)

        # Use cnn to predict probabilities
        test_prediction_raw = cnn.predict(test_data)
        test_prediction_raw = test_prediction_raw / scalar
        test_pred = expit(test_prediction_raw).reshape(-1,)
        test_pred_path = (static_keys[k_test[0]]) + \
            f"_cnn_pred_hybrid_{acc_hz}.npy"
        np.save("saved_outputs/" + test_pred_path, test_pred)

        # save the trained model weights
        cnn.save(cnn_path)


def load_and_train():

    dataset = "stationary"
    static_preprocessed_data = np.load(f'./pre_processed_data/{dataset}/{dataset}_preprocessed_data_{acc_hz}.npy',
                                       allow_pickle=True).item()
    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data)

    dataset = "hybrid"
    hybrid_preprocessed_data = np.load(f'./pre_processed_data/{dataset}/{dataset}_preprocessed_data_{acc_hz}.npy',
                                       allow_pickle=True).item()
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    start_time = time.time()

    train_cnn(static_keys, static_data_bundle, hybrid_data_bundle)
    # train_logreg(static_keys, static_data_bundle)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    load_and_train()
