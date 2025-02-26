
import os
from pathlib import Path


PSG_MASK_VALUE = -1

# use "dyn" for resampling basd on avg Hz per recording
# use a number for fixed resampling
ACC_HZ = "50"  # "50" "dyn" "32"
TARGET_SLEEP = 0.95

ACC_DIFF_GAP = 1.0
SPEC_INPUT_HZ =32 
PSG_DT = 30

# Spectrogram parameters
NFFT=512
WINDOW_LEN=320
NOVERLAP=256
WINDOW='blackman'

N_OUTPUT_EPOCHS = 1024
N_CLASSES = 4
# 2 ** 14 is large enough to hold 8+ hours of 32 Hz spectrogram @ 512 NFFT/256 NOVERLAP
NEW_INPUT_SHAPE = (30 * 512, NFFT // 2, 3)
NEW_OUTPUT_SHAPE = (N_OUTPUT_EPOCHS, N_CLASSES)


# Not exactly constants, but useful "constant" functions
def rgb_path_name(key, saved_model_dir: Path) -> str:
    os.makedirs(saved_model_dir, exist_ok=True)
    return saved_model_dir.joinpath(f"rgb_{key}_{ACC_HZ}.keras")

def rgb_saved_predictions_name(key, saved_output_dir: Path, set_name="static") -> str:
    os.makedirs(saved_output_dir, exist_ok=True)
    return saved_output_dir.joinpath(f"{key}_rgb_pred_{set_name}_{ACC_HZ}.npy")