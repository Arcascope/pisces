from dataclasses import dataclass
from pathlib import Path
import pickle
import zlib

import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import numpy as np
import polars as pl
from pisces.data_sets import DataSetObject
from tqdm import tqdm

# set up plotting to look like ggplot
sns.set_theme('notebook', style='darkgrid')

# Set up constants and locations
DATA_DIR = '/home/eric/Engineering/Work/pisces/data'
FEATURE_COLS = ['ACC_X', 'ACC_Y', 'ACC_Z']
TIMESTAMP_COL = 'TIMESTAMP'
LABEL_COL = 'Sleep_Stage'
NEW_LABEL_COL = 'PSG'
SELECT_COLS = [TIMESTAMP_COL, *FEATURE_COLS, LABEL_COL]
TIMESTAMP_HZ = 64
TIMESTAMP_DT = 1/TIMESTAMP_HZ

ACC_MAX_IDX = 2 ** 22
PSG_MAX_IDX = 2 ** 16

LABEL_MAP = {
    'W': 0,
    'N1': 1,
    'N2': 1,
    'N3': 2,
    'R': 3,
    'Missing': -1,
    'P': 0
}
mapping_df = pl.DataFrame({
    LABEL_COL: list(LABEL_MAP.keys()),
    NEW_LABEL_COL: list(LABEL_MAP.values())
})

@dataclass
class STFT:
    f: np.ndarray
    t: np.ndarray
    Zxx: np.ndarray

    @property
    def shape(self):
        return self.Zxx.shape
    
    def specgram(self, freq_n_tile_clamp: float = 0.0):
        """Produces the absolute value of the STFT, 
        with optional clamping of the values according to
        the given percentile from top/bottom
        """
        abs_array = np.log10(np.abs(self.Zxx) ** 2 + 1e-6)
        if freq_n_tile_clamp > 0:
            abs_array = np.clip(abs_array, 
                                np.percentile(abs_array, freq_n_tile_clamp),
                                np.percentile(abs_array, 100 - freq_n_tile_clamp))
        return abs_array
    
    def plot(self, ax=None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.specgram(freq_n_tile_clamp=0.05), 
                  aspect='auto', origin='lower', extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]])
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        return ax
    
    @classmethod
    def from_acc(cls, 
                 x: np.ndarray, 
                 fs: float = None, 
                 nperseg: int = None, 
                 noverlap: int = None):
        fs = fs or TIMESTAMP_HZ
        nperseg = nperseg or 4 * fs
        noverlap = noverlap or nperseg // 4
        window = signal.windows.hann(nperseg)
        # fft_taker = signal.ShortTimeFFT(win=window, hop=nperseg - noverlap, fs=fs)
        # f, t, Zxx = fft_taker.stft(np.linalg.norm(x, axis=-1))
        f, t, Zxx = signal.stft(np.linalg.norm(x, axis=-1), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

        return cls(f, t, Zxx)


@dataclass
class Preprocessed:
    x: np.ndarray
    y: np.ndarray
    x_spec: STFT = None

    def compute_specgram(self):
        self.x_spec = STFT.from_acc(self.x)



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

        df = data_set.get_feature_data('dfs', d)
        df = df[SELECT_COLS]
        df = df.join(mapping_df, on=LABEL_COL).drop(LABEL_COL)
        x = df[FEATURE_COLS].to_numpy()
        y = df[NEW_LABEL_COL].to_numpy()
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
        prepro_data[d] = Preprocessed(x_pickled_c, y_pickled_c)

    print("Saving preprocessed data")
    np.savez('dreamt_prepro_data.npz', prepro_data)

    print(f"Written to {output_filename}")

def make_beautiful_specgram_plot(prepro_x_y: Preprocessed):
    fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
    if prepro_x_y.x_spec is None:
        prepro_x_y.compute_specgram()
    prepro_x_y.x_spec.plot(ax[0])

    y_plot = prepro_x_y.y
    sns.lineplot(x=np.arange(len(y_plot)), y=y_plot, ax=ax[1])
    ax[1].set_xlim(0, len(y_plot))
    ax[1].set_yticks([-1, 0, 1, 2, 3])
    ax[1].set_yticklabels(['Missing', 'W', 'Light', 'Deep', 'REM'])
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Sleep Stage')
    return fig, ax

if __name__ == '__main__':
    EXCLUDE_THRESHOLD = 18.0
    output_filename = 'dreamt_prepro_data.npz'


    sets = DataSetObject.find_data_sets(DATA_DIR)
    dreamt = sets['dreamt']
    dreamt.parse_data_sets(id_templates='<<ID>>_whole_df.csv')

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
    log_2_max_x = 1
    log_2_max_y = 1

    images_dir = Path(__file__).resolve().parent / 'images'
    images_dir.mkdir(exist_ok=True)
    for k, v in prepro_data.items():
        x = decompress_in_memory(v.x)
        y = decompress_in_memory(v.y)
        print(k, x.shape, y.shape)
        print(f" => {len(x) / TIMESTAMP_HZ / 3600:.1f} hours of data")

        log_len_x = np.log2(len(x))
        log_len_y = np.log2(len(y))
        if log_len_x > log_2_max_x:
            log_2_max_x = log_len_x
        if log_len_y > log_2_max_y:
            log_2_max_y = log_len_y
        
        prepro_data[k] = Preprocessed(x, y)
        fig, ax = make_beautiful_specgram_plot(prepro_data[k])

        print("Saving image to", images_dir)
        plt.savefig(images_dir /  f'{k}_specgram.png', dpi=300)
        plt.close(fig)


    print(f"Max log2(x) = {log_2_max_x:.1f}")
    print(f"Max log2(y) = {log_2_max_y:.1f}")

    

    