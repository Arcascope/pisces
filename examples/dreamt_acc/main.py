from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import time
import zlib

import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import numpy as np
import polars as pl
from examples.dreamt_acc.constants import *
from examples.dreamt_acc.conv2d_net import TrainingResult, make_beautiful_specgram_plot, train_loocv, get_git_commit_hash
from examples.dreamt_acc.preprocess import Preprocessed
from pisces.data_sets import DataSetObject
from tqdm import tqdm

# set up plotting to look like ggplot
sns.set_theme('notebook', style='darkgrid')

# Set up constants and locations
def compress_in_memory(x):
    return zlib.compress(pickle.dumps(x))

def decompress_in_memory(x):
    return pickle.loads(zlib.decompress(x))

def preprocess_data(
        set_ids: list[str],
        data_set: DataSetObject,
        quality_df: pl.DataFrame,
        save_to: str | Path,
        exclude_threshold: float = 18.0):
    prepro_data = {}
    for d in tqdm(set_ids):
        quality_df_filtered = quality_df.filter(pl.col('sid') == d)
        if len(quality_df_filtered) == 0:
            print(f"Skipping {d} due to missing quality data")
            continue
        quality_row = quality_df_filtered.to_dict()
        excluded = 100 * quality_row['percentage_excludes'][0]
        
        if excluded > EXCLUDE_THRESHOLD:
            print(f"Skipping {d} due to {excluded}% > {EXCLUDE_THRESHOLD}% of excludes")
            continue

        # don't keep in memory, easily OOM with the number of subjects
        df = data_set.get_feature_data('dfs', d, keep_in_memory=False)
        df = df[SELECT_COLS]
        df = df.join(mapping_df, on=LABEL_COL).drop(LABEL_COL)
        x = df[FEATURE_COLS].to_numpy()
        y = df[NEW_LABEL_COL].to_numpy()
        del df
        n_pad = TIMESTAMP_HZ - (x.shape[0] % TIMESTAMP_HZ)
        x = np.pad(x, ((0, n_pad), (0, 0)), mode='constant')
        y = np.pad(y, (0, n_pad), mode='constant')
        # x = x.reshape(-1, TIMESTAMP_HZ, 3)
        y = y.reshape(-1, TIMESTAMP_HZ)
        
        # Apply bincount to each row separately
        y_processed = np.zeros(y.shape[0], dtype=int)
        for i in range(y.shape[0]):
            # Compute the most frequent value in this row
            counts = np.bincount(y[i] + 1, minlength=5)
            y_processed[i] = np.argmax(counts) - 1  # undo the +1
        
        y = y_processed  # Replace y with the processed result
        
        print(f"{d} -> {x.shape}, {y.shape}")
        print(f" => {len(x) / TIMESTAMP_HZ / 3600:.1f} hours of data")
        x_pickled_c = compress_in_memory(x)
        y_pickled_c = compress_in_memory(y)
        prepro_data[d] = Preprocessed(d, x_pickled_c, y_pickled_c)
        del x_pickled_c, y_pickled_c

    print("Saving preprocessed data")
    np.savez('dreamt_prepro_data.npz', prepro_data)

    print(f"Written to {output_filename}")


EXAMPLE_DIR = Path(__file__).resolve().parent
EXPERIMENT_RESULTS_CSV = EXAMPLE_DIR / 'dreamt_results.csv'

if __name__ == '__main__':
    EXCLUDE_THRESHOLD = 18.0
    output_filename = 'dreamt_prepro_data.npz'


    sets = DataSetObject.find_data_sets(DATA_DIR)
    dreamt = sets['dreamt']
    dreamt.parse_data(id_templates='<<ID>>_whole_df.csv')

    quality_analysis_file = dreamt.path / 'quality_analysis' / 'quality_scores_per_subject.csv'
    quality_df = pl.read_csv(quality_analysis_file)


    dreamt_ids = dreamt.ids
    # preprocess_data(
    #     dreamt_ids,
    #     dreamt,
    #     quality_df,
    #     output_filename,
    #     EXCLUDE_THRESHOLD
    # )
    prepro_data = np.load('dreamt_prepro_data.npz', allow_pickle=True)['arr_0'].item()

    images_dir = Path(__file__).resolve().parent / 'images'
    images_dir.mkdir(exist_ok=True)
    for k, v in prepro_data.items():
        x = decompress_in_memory(v.x)
        y = decompress_in_memory(v.y)

        
        prepro_k = Preprocessed(v.idno, x, y)
        prepro_k.compute_specgram()
        print(k, prepro_k.x_spec.shape, prepro_k.y.shape)
        if prepro_k.x_spec.shape[0] != PSG_MAX_IDX:
            print(f"Padding {k} to {PSG_MAX_IDX}")
        # print(f" => {len(x) / TIMESTAMP_HZ / 3600:.1f} hours of data")
        prepro_data[k] = prepro_k

        # fig, ax = make_beautiful_specgram_plot(prepro_data[k])

        # print("Saving image to", images_dir)
        # plt.savefig(images_dir /  f'{k}_specgram.png', dpi=300)
        # plt.close(fig)
    
    # stack = np.array([prepro_data[k].x_spec.Zxx for k in prepro_data])

    prepro_values = list(prepro_data.values())
    time_start = time.time()
    results = train_loocv(
        prepro_values, 
        num_epochs=2, 
        batch_size=16,
        experiment_results_csv=EXPERIMENT_RESULTS_CSV,
        lr=1e-3
    )
    time_end = time.time()
    print(f"Training took {time_end - time_start:.1f} seconds // {(time_end - time_start) / 60:.1f} minutes")

    np.savez(f'{get_git_commit_hash()}_dreamt_results.npz', results)
    # results = np.load('dreamt_results.npz', allow_pickle=True)['arr_0'] 

    # for (inputs, outputs) in zip(prepro_values, results):
    #     print(inputs.idno)
    #     fig, ax = make_beautiful_specgram_plot(inputs, outputs)
    #     print("Saving image to", images_dir)
    #     plt.savefig(images_dir /  f'{inputs.idno}_specgram.png', dpi=300)
    #     plt.close(fig)

