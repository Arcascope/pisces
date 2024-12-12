from pathlib import Path
import numpy as np

from examples.RGB_Spectrograms.preprocessing import preprocessed_data_filename



def load_preprocessed_data(dataset: str, preprocessed_data_path: Path = None):
    return np.load(preprocessed_data_filename(dataset, preprocessed_data_path),
                   allow_pickle=True).item()


def print_histogram(data, bins=10):
    """prints a text histogram into the terminal"""
    data = np.array(data)
    hist, bin_edges = np.histogram(data, bins=bins)
    pos_prepend = ""
    if bin_edges[0] < 0:
        pos_prepend = " "
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}: {'#' * hist[i]}")
    print(f"min: {data.min()} max: {data.max()} mean: {data.mean()} std: {data.std()}")
