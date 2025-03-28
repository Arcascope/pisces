from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import time
import zlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from examples.dreamt_acc.constants import *
from examples.dreamt_acc.conv2d_net import TrainingResult, make_beautiful_specgram_plot, get_git_commit_hash
from examples.dreamt_acc.preprocess import Preprocessed
from pisces.data_sets import DataSetObject

def train_eval(train_data_list, test_data_list, experiment_results_csv, num_epochs=20, lr=1e-3, batch_size=1, min_spec_max=0.1, plot=True):
    """Adapts the train_loocv function to train on one dataset and evaluate on another.
    Imports the actual implementation from conv2d_net to avoid circular imports.
    """
    from examples.dreamt_acc.conv2d_net import train_eval as _train_eval
    return _train_eval(
        train_data_list=train_data_list,
        test_data_list=test_data_list,
        experiment_results_csv=experiment_results_csv,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        min_spec_max=min_spec_max,
        plot=plot
    )

# Helper functions from main.py
def compress_in_memory(x):
    return zlib.compress(pickle.dumps(x))

def decompress_in_memory(x):
    return pickle.loads(zlib.decompress(x))

def preprocess_data(set_ids, data_set, quality_df=None, exclude_threshold=18.0):
    """Process data into Preprocessed objects, adapting to handle datasets without quality data"""
    prepro_data = {}
    for d in tqdm(set_ids):
        # Quality filtering if quality data is available
        if quality_df is not None:
            quality_df_filtered = quality_df.filter(pl.col('sid') == d)
            if len(quality_df_filtered) == 0:
                print(f"Skipping {d} due to missing quality data")
                continue
            quality_row = quality_df_filtered.to_dict()
            excluded = 100 * quality_row['percentage_excludes'][0]
            
            if excluded > exclude_threshold:
                print(f"Skipping {d} due to {excluded}% > {exclude_threshold}% of excludes")
                continue

        # Get the feature data
        df = data_set.get_feature_data('dfs', d, keep_in_memory=False)
        df = df[SELECT_COLS]
        df = df.join(mapping_df, on=LABEL_COL).drop(LABEL_COL)
        x = df[FEATURE_COLS].to_numpy()
        y = df[NEW_LABEL_COL].to_numpy()
        del df
        
        # Pad to full timestamp length
        n_pad = TIMESTAMP_HZ - (x.shape[0] % TIMESTAMP_HZ)
        x = np.pad(x, ((0, n_pad), (0, 0)), mode='constant')
        y = np.pad(y, (0, n_pad), mode='constant')
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
        
        # Create Preprocessed object directly instead of compressing first
        prepro_data[d] = Preprocessed(d, x, y)
        prepro_data[d].compute_specgram()

    return prepro_data

def preprocess_walch_data(set_ids, data_set, quality_df=None, exclude_threshold=18.0):
    """Process Walch dataset data into Preprocessed objects.
    The Walch dataset has a different structure with 'accelerometer' and 'psg' features.
    """
    prepro_data = {}
    for d in tqdm(set_ids, desc="Processing Walch subjects"):
        # Quality filtering if quality data is available
        if quality_df is not None:
            quality_df_filtered = quality_df.filter(pl.col('sid') == d)
            if len(quality_df_filtered) == 0:
                print(f"Skipping {d} due to missing quality data")
                continue
            quality_row = quality_df_filtered.to_dict()
            excluded = 100 * quality_row['percentage_excludes'][0]
            
            if excluded > exclude_threshold:
                print(f"Skipping {d} due to {excluded}% > {exclude_threshold}% of excludes")
                continue

        try:
            # Get accelerometer data
            acc_df = data_set.get_feature_data('accelerometer', d, keep_in_memory=False).to_numpy()
            if acc_df is None or len(acc_df) == 0:
                print(f"Skipping {d} - no accelerometer data found")
                continue
                
            # Get PSG data
            psg_df = data_set.get_feature_data('psg', d, keep_in_memory=False).to_numpy()
            if psg_df is None or len(psg_df) == 0:
                print(f"Skipping {d} - no PSG data found")
                continue
                
            x = acc_df[:, 1:4]  # Extract accelerometer data
            y = psg_df[:, 1]  # Extract PSG data
            
            
            n_pad = TIMESTAMP_HZ - (x.shape[0] % TIMESTAMP_HZ)
            x = np.pad(x, ((0, n_pad), (0, 0)), mode='constant')
            y = np.pad(y, (0, PSG_MAX_IDX - y.shape[0]), mode='constant', constant_values=MASK_VALUE)
            
            print(f"{d} -> {x.shape}, {y.shape}")
            print(f" => {len(x) / TIMESTAMP_HZ / 3600:.1f} hours of data")
            
            # Create Preprocessed object
            prepro_data[d] = Preprocessed(d, x, y)
            prepro_data[d].compute_specgram()
            
        except Exception as e:
            print(f"Error processing {d}: {e}")
            continue

    return prepro_data

EXAMPLE_DIR = Path(__file__).resolve().parent
DREAMT_TO_WALCH_CSV = EXAMPLE_DIR / 'dreamt_to_walch_results.csv'
WALCH_TO_DREAMT_CSV = EXAMPLE_DIR / 'walch_to_dreamt_results.csv'
RESULTS_DIR = EXAMPLE_DIR / 'cross_dataset_results'
EXCLUDE_THRESHOLD = 0.18

if __name__ == '__main__':
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load datasets
    print("Finding datasets...")
    sets = DataSetObject.find_data_sets(DATA_DIR)
    
    # 1. Load and preprocess DREAMT dataset
    print("\n=== Processing DREAMT dataset ===")
    dreamt = sets['dreamt']
    dreamt.parse_data(id_templates='<<ID>>_whole_df.csv')
    
    # Load quality data for DREAMT
    quality_analysis_file = dreamt.path / 'quality_analysis' / 'quality_scores_per_subject.csv'
    quality_df = pl.read_csv(quality_analysis_file)
    
    # Check if we have preprocessed DREAMT data
    dreamt_npz = Path('dreamt_prepro_data.npz')
    if dreamt_npz.exists():
        print("Loading preprocessed DREAMT data...")
        dreamt_prepro_data = np.load('dreamt_prepro_data.npz', allow_pickle=True)['arr_0'].item()
        
        # Convert compressed data to Preprocessed objects
        dreamt_preproc = {}
        for k, v in tqdm(dreamt_prepro_data.items(), desc="Processing DREAMT data"):
            if hasattr(v, 'x_spec') and v.x_spec is not None:
                # Already processed
                dreamt_preproc[k] = v
            else:
                # Need to decompress and compute spectrogram
                x = decompress_in_memory(v.x)
                y = decompress_in_memory(v.y)
                prepro_k = Preprocessed(v.idno, x, y)
                prepro_k.compute_specgram()
                dreamt_preproc[k] = prepro_k
    else:
        print("Preprocessing DREAMT data from scratch...")
        dreamt_ids = dreamt.ids
        dreamt_preproc = preprocess_data(dreamt_ids, dreamt, quality_df, EXCLUDE_THRESHOLD)
    
    # 2. Load and preprocess Walch dataset
    print("\n=== Processing Walch dataset ===")
    walch = sets['walch_et_al_64hz']
    walch.parse_data()  # Explicitly specify feature types

    # Check if Walch has quality data; if not, we'll skip that filtering
    walch_quality_file = walch.path / 'quality_analysis' / 'quality_scores_per_subject.csv'
    walch_quality_df = None
    if walch_quality_file.exists():
        walch_quality_df = pl.read_csv(walch_quality_file)

    # Check if we have preprocessed Walch data
    walch_npz = Path('walch_prepro_data.npz')
    if walch_npz.exists():
        print("Loading preprocessed Walch data...")
        walch_prepro_data = np.load('walch_prepro_data.npz', allow_pickle=True)['arr_0'].item()
        
        # Convert compressed data to Preprocessed objects
        walch_preproc = {}
        for k, v in tqdm(walch_prepro_data.items(), desc="Processing Walch data"):
            if hasattr(v, 'x_spec') and v.x_spec is not None:
                # Already processed
                walch_preproc[k] = v
            else:
                # Need to decompress and compute spectrogram
                x = decompress_in_memory(v.x)
                y = decompress_in_memory(v.y)
                prepro_k = Preprocessed(v.idno, x, y)
                prepro_k.compute_specgram()
                walch_preproc[k] = prepro_k
    else:
        print("Preprocessing Walch data from scratch...")
        walch_ids = walch.ids
        # Use the dedicated Walch preprocessing function
        walch_preproc = preprocess_walch_data(walch_ids, walch, walch_quality_df)
        
        # Save preprocessed Walch data
        walch_compressed = {}
        for k, v in tqdm(walch_preproc.items(), desc="Compressing Walch data"):
            # Store compressed versions for potential future use
            x_compressed = compress_in_memory(v.x)
            y_compressed = compress_in_memory(v.y)
            walch_compressed[k] = Preprocessed(v.idno, x_compressed, y_compressed)
        np.savez('walch_prepro_data.npz', walch_compressed)
    
    # 3. Run cross-dataset training and evaluation
    dreamt_values = list(dreamt_preproc.values())
    walch_values = list(walch_preproc.values())
    
    # Train on DREAMT, evaluate on Walch
    print("\n=== Training on DREAMT, testing on Walch ===")
    time_start = time.time()
    dreamt_to_walch_results = train_eval(
        train_data_list=dreamt_values,
        test_data_list=walch_values,
        experiment_results_csv=DREAMT_TO_WALCH_CSV,
        num_epochs=10,
        batch_size=2,
        lr=1e-3
    )
    time_end = time.time()
    print(f"DREAMT→Walch training took {time_end - time_start:.1f} seconds ({(time_end - time_start) / 60:.1f} minutes)")
    
    # Save results
    np.savez(RESULTS_DIR / f'{get_git_commit_hash()}_dreamt_to_walch_results.npz', dreamt_to_walch_results)
    
    # Train on Walch, evaluate on DREAMT
    print("\n=== Training on Walch, testing on DREAMT ===")
    time_start = time.time()
    walch_to_dreamt_results = train_eval(
        train_data_list=walch_values,
        test_data_list=dreamt_values,
        experiment_results_csv=WALCH_TO_DREAMT_CSV,
        num_epochs=10,
        batch_size=2,
        lr=1e-3
    )
    time_end = time.time()
    print(f"Walch→DREAMT training took {time_end - time_start:.1f} seconds ({(time_end - time_start) / 60:.1f} minutes)")
    
    # Save results
    np.savez(RESULTS_DIR / f'{get_git_commit_hash()}_walch_to_dreamt_results.npz', walch_to_dreamt_results)
    
    # Print summary statistics
    print("\n=== Cross-Dataset Performance Summary ===")
    dreamt_to_walch_wasa = np.mean([r.wake_acc for r in dreamt_to_walch_results])
    walch_to_dreamt_wasa = np.mean([r.wake_acc for r in walch_to_dreamt_results])
    
    print(f"DREAMT → Walch: Average WASA: {dreamt_to_walch_wasa:.4f}")
    print(f"Walch → DREAMT: Average WASA: {walch_to_dreamt_wasa:.4f}")
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [
        [r.wake_acc for r in dreamt_to_walch_results],
        [r.wake_acc for r in walch_to_dreamt_results]
    ]
    labels = ['DREAMT → Walch', 'Walch → DREAMT']
    
    ax.boxplot(data, labels=labels)
    ax.set_title('Cross-Dataset Performance (WASA)')
    ax.set_ylabel('Wake Accuracy when Sleep Accuracy = 0.95')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cross_dataset_comparison.png', dpi=300)
    print(f"Saved comparison plot to {RESULTS_DIR / 'cross_dataset_comparison.png'}")