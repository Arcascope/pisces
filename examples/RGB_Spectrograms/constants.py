
# use "dyn" for resampling basd on avg Hz per recording
# use a number for fixed resampling
ACC_HZ = "50"  # "50" "dyn" "32"
TARGET_SLEEP = 0.95

ACC_DIFF_GAP = 1.0
ACC_INPUT_HZ = 50
PSG_DT = 30

N_OUTPUT_EPOCHS = 1024
N_CLASSES = 4
NEW_INPUT_SHAPE = (15360, 256, 3)
NEW_OUTPUT_SHAPE = (N_OUTPUT_EPOCHS, N_CLASSES)