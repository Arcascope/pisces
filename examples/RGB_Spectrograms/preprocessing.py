from typing import Dict
import numpy as np
from examples.RGB_Spectrograms.constants import N_OUTPUT_EPOCHS, NEW_INPUT_SHAPE
import pisces.data_sets as pds
from pisces.utils import accelerometer_to_3d_specgram, build_ADS, pad_or_truncate, resample_accel_data

ACC_DIFF_GAP = 1.0
ACC_INPUT_HZ = 50
PSG_DT = 30


def big_specgram_process(dataset: pds.DataSetObject,
                         subject_id: str,
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
    if ACC_INPUT_HZ == "dyn":
        int_hz = int(avg_time_hz)
        print("dynamic rate:", int_hz)
        sample_rate = int_hz
    else:
        sample_rate = int(ACC_INPUT_HZ)
        print("fixed rate:", sample_rate)

    accel_data_diff = np.diff(accel_data, axis=0)  # take a time diff, this should remove some gravity
    accel_data_diff[:, 0] = accel_data[1:, 0]  # put the time back in
    accel_data_resampled = resample_accel_data(
        accel_data_diff, original_fs=sample_rate, target_fs=ACC_INPUT_HZ)

    # compute spectrogram with resampled data
    spectrograms = accelerometer_to_3d_specgram(
        accel_data_resampled)
    padded_spectrograms = np.zeros(NEW_INPUT_SHAPE)
    padded_spectrograms[:spectrograms.shape[0],
                        ...] = spectrograms[:NEW_INPUT_SHAPE[0], ...]

    # Compute activity with resampled data
    activity_time, ads = build_ADS(accel_data)
    activity_data = np.column_stack((activity_time, ads))

    # Pad PSG data to 1024 samples
    psg_data = pad_or_truncate(psg_data, int(N_OUTPUT_EPOCHS))

    # Pad activity data to 2 * 1024 samples
    activity_data = pad_or_truncate(activity_data, int(N_OUTPUT_EPOCHS * 2))

    return {"spectrogram": padded_spectrograms,
            "activity": activity_data,
            "psg": psg_data}
