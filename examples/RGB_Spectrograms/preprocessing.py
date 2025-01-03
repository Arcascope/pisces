from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy.signal import spectrogram
from keras.layers import AveragePooling2D
import tensorflow as tf
from examples.RGB_Spectrograms.constants import ACC_DIFF_GAP, PSG_MASK_VALUE, SPEC_INPUT_HZ, ACC_HZ, N_OUTPUT_EPOCHS, NEW_INPUT_SHAPE, PSG_DT, NFFT, NOVERLAP, WINDOW_LEN, WINDOW
import pisces.data_sets as pds
from pisces.utils import accelerometer_to_3d_specgram, build_ADS, pad_or_truncate, resample_accel_data
from examples.NHRC.nhrc_utils.analysis import stages_map


DATA_LOCATION = Path('/home/eric/Engineering/Work/pisces/data')

def process_data_set(data_set: pds.DataSetObject,
                     ids_to_exclude: List[str],
                     process_data_fn) -> Dict[str, Dict[str, np.ndarray]]:
    data = {}
    for subject_id in data_set.ids:
        if subject_id in ids_to_exclude:
            continue
        print(f"Processing {subject_id}")
        data[subject_id] = process_data_fn(
            data_set,
            subject_id)
    return data


@dataclass
class PreparedDataRGB:
    spectrograms: np.array
    labels: np.array 
    weights: np.array

    def __getitem__(self, index):
        return PreparedDataRGB(
            spectrograms=self.spectrograms[index][np.newaxis, ...],
            labels=self.labels[index][np.newaxis, ...],
            weights=self.weights[index][np.newaxis, ...]
        )

def compute_weights(labels):
    # TODO: expand to handle multi-class
    # compute class weights here
    n_wake = float(np.sum(labels == 0))
    n_sleep = float(np.sum(labels > 0))
    n_total = n_wake + n_sleep # excludes -1 mask

    class_weights = {
        PSG_MASK_VALUE: 0.0,
        0: (1 / n_wake) * (n_total / 2.0),
        1: (1 / n_sleep) * (n_total / 2.0)
    }
    print("class weights: ", json.dumps(class_weights, indent=2 ))

    weight_stack = np.ones_like(labels, dtype=np.float32)
    for k, v in class_weights.items():
        weight_stack[labels == k] = v
    
    return weight_stack

def preprocessed_set_path(cache_dir: Path, set_name: str) -> Path:
    set_path = cache_dir.joinpath(set_name)
    os.makedirs(set_path, exist_ok=True)
    return set_path

def preprocessed_data_filename(set_name: str, cache_dir: Path | None = None) -> str:
    base_fn = f"{set_name}_preprocessed_data_{ACC_HZ}.npy"
    return base_fn if cache_dir is None else preprocessed_set_path(cache_dir=cache_dir, set_name=set_name).joinpath(base_fn)

def sw_map_fn(x):
    return np.where(x > 0, 1.0, x)

def prepare_data(preprocessed_data, n_classes=4, freq_downsample: int = 1) -> PreparedDataRGB:
    psg_fn = stages_map if n_classes == 4 else sw_map_fn
    label_stack = np.array([
        psg_fn(preprocessed_data[k]['psg'][:, 1])
        for k in list(preprocessed_data.keys())
    ])

    label_weights = compute_weights(label_stack)

    spectrogram_stack = np.array([
        preprocessed_data[k]['spectrogram']
        for k in list(preprocessed_data.keys())
    ])

    # clip the spectrogram stack to 0.05 and 0.95 quantiles
    p_low = 5
    for i in range(spectrogram_stack.shape[-1]):
        p5, p95 = np.percentile(spectrogram_stack[..., i], [p_low, 100 - p_low])
        spectrogram_stack[..., i] = np.clip(spectrogram_stack[..., i], p5, p95)
    
    if freq_downsample > 1:
        # apply avg pooling to downsample the frequency axis
        spectrogram_stack = AveragePooling2D((1, freq_downsample))(spectrogram_stack).numpy()

    return PreparedDataRGB(
        spectrograms=spectrogram_stack.astype(np.float32),
        labels=label_stack.astype(np.float32),
        weights=label_weights.astype(np.float32)
    )

def rgb_specgram(accel_xyz: np.array) -> np.array:
    """
    Convert accelerometer data to spectrograms
    """
    # compute spectrogram with resampled data
    spectrograms, times, freqs = accelerometer_to_3d_specgram(
        accel_xyz,
        nfft=NFFT,
        window_len=WINDOW_LEN,
        noverlap=NOVERLAP,
        window=WINDOW,
        fs=SPEC_INPUT_HZ)
    padded_spectrograms = np.zeros(NEW_INPUT_SHAPE)
    padded_spectrograms[:spectrograms.shape[0],
                        ...] = spectrograms[:NEW_INPUT_SHAPE[0], ...]
    return padded_spectrograms, times, freqs

def norm_specgram(accel_xyz: np.array) -> np.array:
    """
    Take L2 norm of xyz, then take a specgram
    """
    accel_xyz_norm = np.linalg.norm(accel_xyz[:, 1:], axis=1)
    f, t, Sxx = spectrogram(
        accel_xyz_norm, 
        fs=SPEC_INPUT_HZ,  # Sampling frequency after resampling
        nfft=NFFT, 
        nperseg=WINDOW_LEN, 
        noverlap=NOVERLAP, 
        window=WINDOW)
    # trim off the last frequency bin, which is probably uninteresting
    spectrograms = Sxx[:-1].T
    spectrograms = np.expand_dims(spectrograms, axis=-1)
    NORM_INPUT_SHAPE = [*NEW_INPUT_SHAPE[:-1], 1]
    padded_spectrograms = np.zeros(NORM_INPUT_SHAPE)
    padded_spectrograms[:spectrograms.shape[0],
                        ...] = spectrograms[:NORM_INPUT_SHAPE[0], ...]
    return padded_spectrograms, t, f
    # spectrograms.append(Sxx.T)  # Transpose to shape (time_bins, freq_bins)
    # times.append(t)
    # frequencies.append(f)

    # accel_xyz_norm = np.column_stack((accel_xyz[:, 0], accel_xyz_norm))
    # return rgb_specgram(accel_xyz_norm)

def big_specgram_process(dataset: pds.DataSetObject,
                         subject_id: str,
                         xyz_accel_to_specgram_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
                         *args, **kwargs) -> Dict[str, np.ndarray]:
    accel_data = dataset.get_feature_data(
        'accelerometer', subject_id).to_numpy()
    psg_data = dataset.get_feature_data('psg', subject_id).to_numpy()

    # Sort based on time (axis 0)
    accel_data = accel_data[accel_data[:, 0].argsort()]
    psg_data = psg_data[psg_data[:, 0].argsort()]

    # Convert activity and PSG time to int
    psg_data[:, 0] = np.round(psg_data[:, 0])

    # Trim data to common time range
    start_time = max(accel_data[0, 0], psg_data[0, 0])
    end_time = min(accel_data[-1, 0], psg_data[-1, 0])

    accel_data = accel_data[(accel_data[:, 0] >= start_time)
                            & (accel_data[:, 0] <= end_time)]
    psg_data = psg_data[(psg_data[:, 0] >= start_time)
                        & (psg_data[:, 0] <= end_time)]

    # Find gaps in accelerometer data
    time_diff = np.diff(accel_data[:, 0])
    avg_time_hz = int(1/np.median(time_diff))
    gap_indices = np.where(time_diff > ACC_DIFF_GAP)[0]

    # Mask PSG labels during accelerometer gaps
    pre_mask_sleeps = np.sum(psg_data[:, 1] > 0)
    pre_mask_wakes = np.sum(psg_data[:, 1] == 0)
    wakes_masked = 0

    for gap_index in gap_indices:
        gap_start = accel_data[gap_index, 0] + ACC_DIFF_GAP
        gap_end = accel_data[gap_index + 1, 0]
        mask_indices = np.where(
            (psg_data[:, 0] + PSG_DT >= gap_start) &
            (psg_data[:, 0] <= gap_end))[0]
        wake_counts = np.sum((psg_data[mask_indices, 1].astype(int)) == 0)
        wakes_masked += wake_counts
        psg_data[mask_indices, 1:] = -1
    
    print_class_statistics(psg_data[:, 1])

    post_mask_sleeps = np.sum(psg_data[:, 1] > 0)
    post_mask_wakes = np.sum(psg_data[:, 1] == 0)

    # Convert accelerometer data to spectrograms

    sample_rate = 50  # we configure this next
    if ACC_HZ == "dyn":
        int_hz = int(avg_time_hz)
        print("dynamic rate:", int_hz)
        sample_rate = int_hz
    else:
        sample_rate = int(ACC_HZ)
        print("fixed rate:", sample_rate)

    accel_data_resampled = resample_accel_data(
        accel_data, original_fs=sample_rate, target_fs=SPEC_INPUT_HZ)
    accel_data_diff = np.zeros_like(accel_data_resampled[1:])
    accel_data_diff[:, 1:] = np.diff(accel_data_resampled[:, 1:], axis=0)  # take a time diff, this should remove some gravity
    accel_data_diff[:, 0] = accel_data_resampled[1:, 0]  # put the time back in

    # compute spectrogram with resampled data
    spectrograms, times, freqs = rgb_specgram(accel_data_resampled) \
        if xyz_accel_to_specgram_fn is None \
            else xyz_accel_to_specgram_fn(accel_data_resampled)
    # accelerometer_to_3d_specgram(
    #     accel_data_diff,
    #     nfft=NFFT,
    #     window_len=WINDOW_LEN,
    #     noverlap=NOVERLAP,
    #     window=WINDOW,
    #     fs=SPEC_INPUT_HZ)
    # padded_spectrograms = np.zeros(NEW_INPUT_SHAPE)
    # padded_spectrograms[:spectrograms.shape[0],
    #                     ...] = spectrograms[:NEW_INPUT_SHAPE[0], ...]

    # Compute activity with resampled data
    activity_time, ads = build_ADS(accel_data_resampled)
    activity_data = np.column_stack((activity_time, ads))

    # Pad PSG data to 1024 samples
    psg_data = pad_or_truncate(psg_data, int(N_OUTPUT_EPOCHS))

    # Pad activity data to 2 * 1024 samples
    activity_data = pad_or_truncate(activity_data, int(N_OUTPUT_EPOCHS * 2))


    return {"spectrogram": spectrograms,
            "spec_times": times,
            "spec_freqs": freqs,
            "activity": activity_data,
            "psg": psg_data}


def print_class_statistics(labels: np.array):
    """
    Print the % of samples in each class
    """
    print("Class statistics:")
    values, *_, counts = np.unique(labels, return_counts=True)
    print("MOST COMMON CLASS:", values[np.argmax(counts)], f"at {counts[np.argmax(counts)]/len(labels)*100:.2f}%")
    print(f"SLEEP %: {np.sum(labels > 0) / len(labels) * 100:.2f}")


def do_preprocessing(process_data_fn=None, cache_dir: Path | str | None = None) -> float:
    # clean_and_save_accelerometer_data()

    start_run = time.time()
    print("data_location: ", DATA_LOCATION)

    sets = pds.DataSetObject.find_data_sets(DATA_LOCATION)
    walch = sets['walch_et_al']
    walch.parse_data_sets()
    print(f"Found {len(walch.ids)} subjects")

    hybrid = sets['hybrid_motion']
    hybrid.parse_data_sets()
    print(f"Found {len(hybrid.ids)} subjects")

    subjects_to_exclude_walch = [
        "7749105",
        "5383425",
        "8258170"
    ]

    subjects_to_exclude_hybrid = subjects_to_exclude_walch

    # Process the datasets
    print("PROCESSING WALCH DATA")
    preprocessed_data_walch = process_data_set(
        walch, subjects_to_exclude_walch, process_data_fn)
    print("PROCESSING HYBRID DATA")
    preprocessed_data_hybrid = process_data_set(
        hybrid, subjects_to_exclude_hybrid, process_data_fn)
    print("DONE PROCESSING")

    CWD = Path(os.getcwd())
    save_path = CWD.joinpath("pre_processed_data") if cache_dir is None else Path(cache_dir)

    hybrid_name = "hybrid"
    stationary_name = "stationary"

    hybrid_path = preprocessed_set_path(save_path, hybrid_name)
    os.makedirs(hybrid_path, exist_ok=True)

    walch_path = preprocessed_set_path(save_path, stationary_name)
    os.makedirs(walch_path, exist_ok=True)

    save_preprocessing_to = preprocessed_data_filename(stationary_name, save_path)
    print(f"Saving to {save_preprocessing_to}...")
    with open(save_preprocessing_to, 'wb') as f:
        np.save(f, preprocessed_data_walch)

    save_preprocessing_to =  preprocessed_data_filename(hybrid_name, save_path)
    print(f"Saving to {save_preprocessing_to}...")
    with open(save_preprocessing_to, 'wb') as f:
        np.save(f, preprocessed_data_hybrid)

    end_run = time.time()
    time_taken = end_run - start_run
    print(f"Preprocessing took {end_run - start_run} seconds")
    return time_taken

def prepared_labels_tf(labels: tf.Tensor, mask_value: int = -1, mask_to: int | None = None, pos_class: int = 1) -> np.array:
    prep_labels = tf.where(labels == pos_class, 1, 0)

    # numpy is this: prep_labels[labels == mask_value] = mask_to if mask_to is not None else mask_value
    prep_labels = tf.where(labels == mask_value, mask_to if mask_to is not None else mask_value, prep_labels)

    return prep_labels

def prepared_labels(labels: np.array, mask_value: int = -1, mask_to: int | None = None, pos_class: int = 1) -> np.array:
    prep_labels = np.where(labels == pos_class, 1, 0)
    prep_labels[labels == mask_value] = mask_to if mask_to is not None else mask_value

    return prep_labels
