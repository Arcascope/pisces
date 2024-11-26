
# use "dyn" for resampling basd on avg Hz per recording
# use a number for fixed resampling
ACC_HZ = "50"  # "50" "dyn" "32"
TARGET_SLEEP = 0.95


N_OUTPUT_EPOCHS = 1024
N_CLASSES = 4
NEW_INPUT_SHAPE = (15360, 257, 3)
NEW_OUTPUT_SHAPE = (N_OUTPUT_EPOCHS, N_CLASSES)