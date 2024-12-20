from pathlib import Path
import numpy as np

from examples.RGB_Spectrograms.preprocessing import preprocessed_data_filename



def load_preprocessed_data(dataset: str, preprocessed_data_path: Path = None):
    return np.load(preprocessed_data_filename(dataset, preprocessed_data_path),
                   allow_pickle=True).item()


def print_histogram(data, bins: int=10):
    """prints a text histogram into the terminal"""
    data = np.array(data)
    bins_arr = np.linspace(0, 1, bins+1)
    hist, bin_edges = np.histogram(data, bins=bins_arr)
    pos_prepend = ""
    if bin_edges[0] < 0:
        pos_prepend = " "
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}: {'#' * hist[i]}")
    print(f"min: {data.min():0.3f} max: {data.max():0.3f} mean: {data.mean():0.3f} std: {data.std():0.3f}")
