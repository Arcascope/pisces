import json
from dataclasses import dataclass
from pathlib import Path
import sys

import keras
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf

from analyses.NHRC.nhrc_utils.model_definitions import LR_POST_PAD, LR_PRE_PAD, EXTRA_LOWER, LR_LOWER
from pisces.models import load_saved_keras

ID_COLUMN = 'test_id'
THRESHOLD = 'threshold'
SCENARIO_COLUMN = 'scenario'
MODEL_COLUMN = 'model'
SLEEP_ACCURACY_COLUMN = 'sleep_accuracy'
WASA_SLEEP_ACCURACY = [
    0.93, 0.95, 0.97
]
STATIONARY_LOWER = "stationary"
HYBRID_LOWER = "hybrid"
SCENARIOS = [
    STATIONARY_LOWER,
    HYBRID_LOWER,
]
WASA_COLUMN = 'WASA'
AUROC_COLUMN = "AUROC"
ACCURACY_COLUMN = "TST_Error"
EVALUATION_COLUMNS = [AUROC_COLUMN, ACCURACY_COLUMN, WASA_COLUMN,  THRESHOLD, SLEEP_ACCURACY_COLUMN]

# When we analyze the outcomes of our training, 
# we'll stick the data into a pandas dataframe with these columns.
DF_COLUMNS = [ID_COLUMN, SCENARIO_COLUMN, MODEL_COLUMN, *EVALUATION_COLUMNS]

# Define where we will save the components of our analysis
DEFAULT_EVALUATION_DIR = Path(__file__).parent.parent.joinpath("evaluations")
DEFAULT_EVALUATION_DF_PATH = DEFAULT_EVALUATION_DIR.joinpath("evaluation_df.csv")
DEFAULT_EVALUATION_DIR.parent.mkdir(parents=True, exist_ok=True)

def make_model_filename(excluded_id: str, model: str, evaluation_dir: Path = DEFAULT_EVALUATION_DIR) -> str:
    return evaluation_dir.joinpath("models").joinpath(f"{model}_{excluded_id}.keras")

def make_lr_filename(excluded_id: str, evaluation_dir: Path = DEFAULT_EVALUATION_DIR) -> str:
    return make_model_filename(excluded_id, LR_LOWER, evaluation_dir)

def make_finetuning_filename(excluded_id: str, evaluation_dir: Path = DEFAULT_EVALUATION_DIR) -> str:
    return make_model_filename(excluded_id, EXTRA_LOWER, evaluation_dir)

def load_model(excluded_id: str, model: str, evaluation_dir: Path = DEFAULT_EVALUATION_DIR) -> keras.Model:
    model_filename = make_model_filename(excluded_id, model, evaluation_dir)
    return keras.models.load_model(model_filename)

def load_evaluation_df(evaluation_dir: Path = DEFAULT_EVALUATION_DF_PATH) -> pd.DataFrame:
    return pd.read_csv(evaluation_dir)

def compute_sample_weights(labels: np.ndarray, verbose: bool = False) -> np.ndarray:
    mask_weights = labels >= 0
    if verbose:
        print(f"Scored % of {labels.size} epochs:\n{100 * (np.sum(mask_weights) / mask_weights.size):.2f} ({np.sum(mask_weights)} / {mask_weights.size})")

    class_weight_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=labels[mask_weights].flatten()
    )
    class_weights = {0: class_weight_array[0], 1: class_weight_array[1]}
    if verbose:
        print("Class weights:\n", json.dumps(class_weights, indent=2)) # pretty print
    class_weights |= {-1: 0.0}

    def weight_fn(x):
        return class_weights[x]
    
    wfv = np.vectorize(weight_fn)
    sample_weights = wfv(labels)

    return sample_weights

def compute_mae_for_sleep_time(y_true: tf.Tensor | np.ndarray, y_pred: tf.Tensor | np.ndarray, weights: tf.Tensor | np.ndarray = None) -> float:
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(weights, tf.Tensor):
        weights = weights.numpy()
    # Reshape to simplify processing (remove the singleton dimensions)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    # Mask out the indices where y_true == -1
    include_sel = weights > 0
    y_true = y_true[include_sel]
    y_pred = y_pred[include_sel]
    
    # Calculate total sleep time (sum of values where label is 1)
    SECONDS_PER_PSG = 30
    SECONDS_PER_MINUTE = 60
    SCALAR = SECONDS_PER_PSG / SECONDS_PER_MINUTE
    
    true_sleep_time = np.sum(y_true == 1)
    pred_sleep_time = np.sum(y_pred == 1)
    # Compute Mean Absolute Error (MAE) for total sleep time
    # mae = tf.abs(SCALAR * (true_sleep_time - pred_sleep_time))
    mae = SCALAR * (true_sleep_time - pred_sleep_time)

    print(f"True sleep time: {true_sleep_time} minutes, predicted {pred_sleep_time} minutes (%E: {mae / true_sleep_time:.2f})")
    
    
    return mae



def masked_weighted_accuracy(y_true, y_pred, sample_weight):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sample_weight = tf.cast(sample_weight, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    weighted_accuracy = tf.reduce_sum(correct * sample_weight) / tf.reduce_sum(sample_weight)

    return weighted_accuracy.numpy()

def find_best_threshold(y_true, y_pred, weights, sleep_accuracy, tol: bool = 1e-4):

    threshold_min = 0
    threshold_max = 1
    best_threshold = (threshold_max + threshold_min) / 2
    true_sleep = (y_true == 1) & (weights > 0)
    n_sleep = np.sum(true_sleep)

    just_sleep_pred = y_pred[true_sleep]
    while threshold_max - threshold_min > sys.float_info.epsilon:
        threshold = (threshold_max + threshold_min) / 2
        # only have elements where y_true == 1, i.e. real sleep to score
        y_pred_bin = (just_sleep_pred >= threshold).astype(int)
        # just summing up the 1s, which are "sleep predicted & sleep true"
        accuracy = np.sum(y_pred_bin) / n_sleep
        if accuracy < sleep_accuracy:
            threshold_max = threshold
        else:
            threshold_min = threshold
        if abs(accuracy - sleep_accuracy) < tol:
            break
    best_threshold = (threshold_max + threshold_min) / 2
    print(f"Threshold: {best_threshold}, Accuracy: {accuracy}")

    return best_threshold

def auroc_balaccuracy_wasa(split_name, binary_pred_proba, test_sample_weights, test_labels_masked, sleep_accuracy, verbose: bool = False):
    if isinstance(sleep_accuracy, list):
        return [
            auroc_balaccuracy_wasa(
                f"{split_name}_{int(sleep_accuracy[i] * 100)}",
                binary_pred_proba,
                test_sample_weights,
                test_labels_masked,
                sleep_accuracy[i],
                verbose
            )
            for i in range(len(sleep_accuracy))
        ]

    flat_test_labels = test_labels_masked.reshape(-1,)
    flat_weights = test_sample_weights.reshape(-1,)

    # # compute AUROC
    fpr, tpr, thresholds = roc_curve(flat_test_labels, binary_pred_proba, sample_weight=flat_weights)
    roc_auc = auc(fpr, tpr)

    # # compute WASA
    wasa_threshold = find_best_threshold(
        flat_test_labels,
        binary_pred_proba,
        flat_weights,
        sleep_accuracy
    )#thresholds[np.sum(tpr <= sleep_accuracy)] - 1e-7
    y_guess = binary_pred_proba >= wasa_threshold
    guess_right = y_guess == flat_test_labels
    y_wake = (flat_test_labels == 0) & (flat_weights > 0)
    n_wake = np.sum(y_wake)
    n_wake_right = np.sum(y_wake & guess_right)
    wake_accuracy = n_wake_right / n_wake

    # weighted_accuracy = masked_weighted_accuracy(
    #     flat_test_labels,
    #     y_guess,
    #     flat_weights)
    mae_tst = compute_mae_for_sleep_time(flat_test_labels, y_guess, weights=flat_weights)

    if verbose:
        print("=" * 20)
        print(f"Fold {split_name}")
        print(f"\tROC AUC: {roc_auc}")
        # print(f"\tWEIGHTED accuracy @ ≧{WASA_SLEEP_PERCENT}% sleep: {mae_tst}")
        print(f"\tTotal Sleep Time: {mae_tst}")
        print(f"\t{WASA_COLUMN}: {wake_accuracy} ({n_wake_right} / {n_wake} || threshold: {wasa_threshold})")
        print("=" * 20)
    return roc_auc, mae_tst, wake_accuracy, wasa_threshold


@dataclass
class PreparedData:
    activity: tf.Tensor
    spectrogram: tf.Tensor
    mo_predictions: tf.Tensor
    true_labels: tf.Tensor
    sample_weights: tf.Tensor

mo_keras = load_saved_keras()
def mo_predict_logits(data: tf.Tensor):
    return mo_keras.predict(data)

def prepare_data(preprocessed_data: dict):
    keys = list(preprocessed_data.keys())
    xyz_specgram_input = np.array([
        preprocessed_data[k]['spectrogram']
        for k in keys
    ])
    xyz_average = xyz_specgram_input #np.mean(xyz_specgram_input, axis=-1)

    specgram_input = np.zeros((*xyz_average.shape, 2))
    # embed x,y,z into 2 channels
    # one is reflected along the frequency axis
    specgram_input[..., 0] = xyz_average
    specgram_input[..., 1] = xyz_average[..., ::-1]
    
    mo_predictions = mo_predict_logits(specgram_input)

    full_labels = np.array([
        preprocessed_data[k]['psg'][:, 1]
        for k in keys
    ])
    labels = np.where(full_labels > 0, 1, full_labels)

    # in original setup, specgrams were average of x,y,z
    # specgrams = np.mean(xyz_specgram_input, axis=-1)
    specgrams = xyz_specgram_input

    activity = np.array([
        preprocessed_data[k]['activity'][:, 1]
        for k in keys
    ])
    activity = np.pad(activity, ((0, 0), (LR_PRE_PAD, LR_POST_PAD)), mode='constant', constant_values=0)
    weights = np.array([
        compute_sample_weights(labels[i])
        for i in range(labels.shape[0])
    ])
    return PreparedData(
        activity=tf.convert_to_tensor(activity, dtype=tf.float32),
        spectrogram=tf.convert_to_tensor(specgrams, dtype=tf.float32),
        mo_predictions=tf.convert_to_tensor(mo_predictions, dtype=tf.float32),
        true_labels=tf.convert_to_tensor(labels, dtype=tf.float32),
        sample_weights=tf.convert_to_tensor(weights, dtype=tf.float32)
    )