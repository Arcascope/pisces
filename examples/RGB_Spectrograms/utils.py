from pathlib import Path
import numpy as np

from examples.RGB_Spectrograms.preprocessing import preprocessed_data_filename



def load_preprocessed_data(dataset: str, preprocessed_data_path: Path = None):
    return np.load(preprocessed_data_filename(dataset, preprocessed_data_path),
                   allow_pickle=True).item()
