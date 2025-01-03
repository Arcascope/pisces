
from pathlib import Path
import numpy as np
import tensorflow as tf

from examples.RGB_Spectrograms.plotting import create_histogram_rgb
from examples.RGB_Spectrograms.preprocessing import big_specgram_process, do_preprocessing, norm_specgram
from examples.RGB_Spectrograms.rgb_segmentation import load_and_train

local_dir = Path(__file__).resolve().parent
print("local_dir: ", local_dir)

from functools import partial

norm_big_specgram_process = partial(big_specgram_process, xyz_accel_to_specgram_fn = norm_specgram)

if __name__ == "__main__":
    import warnings

    # set the random seed to 20241217
    SEED = 20241217
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    preprocessed_data_path = local_dir.joinpath("pre_processed_data")
    predictions_path = local_dir.joinpath("saved_outputs")

    total_time = 0
    # total_time += do_preprocessing(big_specgram_process, cache_dir=preprocessed_data_path)
    # total_time += do_preprocessing(norm_big_specgram_process, cache_dir=preprocessed_data_path)
    total_time += load_and_train(
        preprocessed_path=preprocessed_data_path, 
        epochs=35,  # 37 is eyeballed from TesnorBoard
        batch_size=1, 
        lr=5e-5, 
        use_logits=True, 
        n_classes=2,
        sleep_proba=True,
        predictions_path=predictions_path)
    total_time += create_histogram_rgb(
        "rgb", 
        preprocessed_data_path=preprocessed_data_path,
        saved_output_dir=predictions_path,
        sleep_proba=True)
    print(f"TOTAL Total time taken: {total_time:.2f} seconds")


# 0. focus on sleep wake
# 1. trim input size down (either downsample img in-network or pre-network)
# 2. increase number of easy wake epochs
# 3. explore more input transformations: 
#   i. xyz amplitude before specgram (vs learning RGB + rotations)
#   ii. different specgram types (different parameters)