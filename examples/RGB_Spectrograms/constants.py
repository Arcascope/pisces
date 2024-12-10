
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