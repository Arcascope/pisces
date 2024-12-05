import os
import time
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import ACC_HZ as acc_Hz_str

import pisces.data_sets as pds
from pisces.data_sets import (
    DataSetObject,
    ModelInputSpectrogram,
    ModelOutputType,
    DataProcessor,
    PSGType
)

from pisces.utils import build_ADS, pad_or_truncate, resample_accel_data

from examples.NHRC.nhrc_utils.model_definitions import LR_ACTIVITY_INPUTS


FIXED_LABEL_LENGTH = 1024
FIXED_SPECGRAM_SHAPE = (15360, 32)

ACC_DIFF_GAP = 10  # seconds
ACC_RAW_HZ = 50
ACC_RAW_DT = 1/ACC_RAW_HZ
ACC_INPUT_HZ = 32
ACTIVITY_DT = 15
ACTIVITY_HZ = 1/ACTIVITY_DT
PSG_DT = 30
PSG_HZ = 1/PSG_DT
SECONDS_PER_KERNEL = 5 * 60
ACTIVITY_KERNEL_WIDTH = SECONDS_PER_KERNEL * ACTIVITY_HZ
ACTIVITY_KERNEL_WIDTH += 1 - (ACTIVITY_KERNEL_WIDTH % 2)  # Ensure it is odd
# DATA_LOCATION = Path('/Users/ojwalch/Documents/eric-pisces/datasets')
# DATA_LOCATION = Path('/Users/eric/Engineering/Work/pisces/data')
DATA_LOCATION = Path('/home/eric/Engineering/Work/pisces/data')


def clean_and_save_accelerometer_data():
    '''Used to convert raw acceleration from PhysioNet into CSV format'''
    walch_et_al_dir = DATA_LOCATION.joinpath('walch_et_al')
    input_dir = walch_et_al_dir.joinpath('cleaned_accelerometer')
    output_dir = walch_et_al_dir.joinpath('processed_accelerometer')
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob('*_acceleration.txt'):
        df = pd.read_csv(file, delim_whitespace=True,
                         header=None, names=['Timestamp', 'x', 'y', 'z'])
        df = df[df['Timestamp'] >= 0]
        output_file = output_dir / f"{file.stem.split('_')[0]}.csv"
        df.to_csv(output_file, index=False)


def process_data(dataset: pds.DataSetObject,
                 processor: DataProcessor,
                 subject_id) -> Dict[str, np.ndarray]:

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

    print("Pre-mask:\n\tSleeps", pre_mask_sleeps, "\n\tWakes", pre_mask_wakes)
    for gap_index in gap_indices:
        gap_start = accel_data[gap_index, 0] + ACC_DIFF_GAP
        gap_end = accel_data[gap_index + 1, 0]
        mask_indices = np.where(
            (psg_data[:, 0] + PSG_DT >= gap_start) &
            (psg_data[:, 0] <= gap_end))[0]
        wake_counts = np.sum((psg_data[mask_indices, 1].astype(int)) == 0)
        wakes_masked += wake_counts
        psg_data[mask_indices, 1:] = -1

    post_mask_sleeps = np.sum(psg_data[:, 1] > 0)
    post_mask_wakes = np.sum(psg_data[:, 1] == 0)
    print("Post-mask:\n\tSleeps", post_mask_sleeps, "\n\tWakes",
          post_mask_wakes, "\n\tWakes masked", wakes_masked)

    # Convert accelerometer data to spectrograms

    sample_rate = 50
    if acc_Hz_str == "dyn":
        int_hz = int(avg_time_hz)
        print("dynamic rate:", int_hz)
        sample_rate = int_hz
    else:
        sample_rate = int(acc_Hz_str)
        print("fixed rate:", sample_rate)
    accel_data_resampled = resample_accel_data(
        accel_data, original_fs=sample_rate, target_fs=ACC_INPUT_HZ)

    # compute spectrogram with resampled data
    spectrograms = processor.accelerometer_to_spectrogram(
        accel_data_resampled)
    padded_spectrograms = np.zeros(FIXED_SPECGRAM_SHAPE)
    padded_spectrograms[:spectrograms.shape[0],
                        ...] = spectrograms[:FIXED_SPECGRAM_SHAPE[0], ...]

    # Compute activity with resampled data
    activity_time, ads = build_ADS(accel_data)
    activity_data = np.column_stack((activity_time, ads))

    # Pad PSG data to 1024 samples
    psg_data = pad_or_truncate(psg_data, int(FIXED_LABEL_LENGTH))

    # Pad activity data to 2 * 1024 samples
    activity_data = pad_or_truncate(activity_data, int(LR_ACTIVITY_INPUTS))

    return {"spectrogram": padded_spectrograms,
            "activity": activity_data,
            "psg": psg_data}



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




def do_preprocessing(process_data_fn=None, cache_dir: Path | str | None = None):
    # clean_and_save_accelerometer_data()

    start_run = time.time()
    print("data_location: ", DATA_LOCATION)

    sets = DataSetObject.find_data_sets(DATA_LOCATION)
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

    input_features = ['accelerometer']
    model_input = ModelInputSpectrogram(input_features, 32)
    output_type = ModelOutputType.WAKE_LIGHT_DEEP_REM
    data_processor_walch = DataProcessor(walch,
                                         model_input,
                                         output_type=output_type,
                                         psg_type=PSGType.HAS_N4)
    if process_data_fn is None:
        def process_data_fn(data_set, subjects_to_exclude):
            return process_data(data_set, data_processor_walch, subjects_to_exclude, )

    # Process the datasets
    preprocessed_data_walch = process_data_set(
        walch, subjects_to_exclude_walch, process_data_fn)
    preprocessed_data_hybrid = process_data_set(
        hybrid, subjects_to_exclude_hybrid, process_data_fn)

    CWD = Path(os.getcwd())
    save_path = CWD.joinpath("pre_processed_data") if cache_dir is None else Path(cache_dir)

    hybrid_name = "hybrid"
    stationary_name = "stationary"

    hybrid_path = save_path.joinpath(hybrid_name)
    os.makedirs(hybrid_path, exist_ok=True)

    walch_path = save_path.joinpath(stationary_name)
    os.makedirs(walch_path, exist_ok=True)

    save_preprocessing_to = walch_path.joinpath(
        f"{stationary_name}_preprocessed_data_{acc_Hz_str}.npy")
    print(f"Saving to {save_preprocessing_to}...")
    with open(save_preprocessing_to, 'wb') as f:
        np.save(f, preprocessed_data_walch)

    save_preprocessing_to = hybrid_path.joinpath(
        f"{hybrid_name}_preprocessed_data_{acc_Hz_str}.npy")
    print(f"Saving to {save_preprocessing_to}...")
    with open(save_preprocessing_to, 'wb') as f:
        np.save(f, preprocessed_data_hybrid)

    end_run = time.time()
    print(f"Preprocessing took {end_run - start_run} seconds")


if __name__ == "__main__":
    do_preprocessing()
