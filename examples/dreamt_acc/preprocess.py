import time

from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import polars as pl
from tqdm import tqdm

from examples.dreamt_acc.constants import MASK_VALUE, PSG_MAX_IDX, TIMESTAMP_HZ

from pisces.data_sets import DataSetObject


@dataclass
class STFT:
    f: np.ndarray
    t: np.ndarray
    Zxx: np.ndarray
    specgram: np.ndarray = None

    @property
    def shape(self):
        return self.Zxx.shape
    
    def compute_specgram(self,
                         freq_min: float | None = None,
                         freq_max: float | None = None,
                         n_tile_clamp: float = 0.05,
                         normalization_window_idx: int | None = None) -> np.ndarray:
        """Produces the absolute value of the STFT, 
        with optional clamping of the values according to
        the given percentile from top/bottom
        """
        abs_array = np.log10(np.abs(self.Zxx) ** 2 + 1e-6)
        self.specgram = abs_array
        if n_tile_clamp > 0:
            self.specgram = np.clip(self.specgram, 
                                np.percentile(self.specgram, n_tile_clamp),
                                np.percentile(self.specgram, 100 - n_tile_clamp))
        if normalization_window_idx is not None:
            self.apply_local_stdnorm_to_specgram(normalization_window_idx)
        if freq_min is not None:
            f_select = self.f >= freq_min
            self.specgram = self.specgram[:, f_select]
            self.f = self.f[f_select]
            self.Zxx = self.Zxx[:, f_select]
        if freq_max is not None:
            f_select = self.f <= freq_max
            self.specgram = self.specgram[:, f_select]
            self.f = self.f[f_select]
            self.Zxx = self.Zxx[:, f_select]
        return abs_array
    
    def apply_local_stdnorm_to_specgram(self, window_size: int = 5) -> np.ndarray:
        """Applies local standardization to the specgram
        """
        means_array = np.zeros_like(self.specgram)
        stdevs_array = np.zeros_like(self.specgram)
        for i in range(self.specgram.shape[0]):
            specgram_window = self.specgram[max(0, i - window_size):i + window_size, :]
            means_array[i, :] = np.mean(specgram_window, axis=0)
            stdevs_array[i, :] = np.std(specgram_window, axis=0) + 1e-6
        self.specgram = (self.specgram - means_array) / stdevs_array
        return self.specgram

    
    def plot(self, ax=None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if self.specgram is None:
            self.compute_specgram()
        ax.imshow(self.specgram.T, 
                  aspect='auto', origin='lower', 
                  extent=[0, len(self.t), self.f[0], self.f[-1]],
                  vmin=-6.1, vmax=0.75
                  )
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        return ax
    
    @classmethod
    def from_acc(cls, 
                 x: np.ndarray, 
                 fs: float = None, 
                 nperseg: int = None, 
                 noverlap: int = None) -> 'STFT':
        fs = fs or TIMESTAMP_HZ
        nperseg = nperseg or 4 * fs
        noverlap = nperseg - fs
        window = signal.windows.hann(nperseg)
        # fft_taker = signal.ShortTimeFFT(win=window, hop=nperseg - noverlap, fs=fs)
        # f, t, Zxx = fft_taker.stft(np.linalg.norm(x, axis=-1))
        x_norm = np.linalg.norm(x, axis=-1)
        x_diff = np.diff(x_norm)
        f, t, Zxx = signal.stft(x_diff, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

        return cls(f, t, Zxx.T)


@dataclass
class Preprocessed:
    idno: str
    x: np.ndarray
    y: np.ndarray
    x_spec: STFT = None

    def compute_specgram(self, pad_to_psg_max_idx: bool = True):
        self.x_spec = STFT.from_acc(self.x)
        if pad_to_psg_max_idx:
            self.pad_to_psg_max_idx()
    
    def pad_to_psg_max_idx(self):
        if self.x_spec is None:
            return
        
        if self.x_spec.shape[0] <= PSG_MAX_IDX:
            self.x_spec.Zxx = np.pad(
                self.x_spec.Zxx,
                ((0, PSG_MAX_IDX - self.x_spec.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0
            )
            # extend the time axis
            self.x_spec.t = np.linspace(
                self.x_spec.t[0],
                # not / TIMESTAMP_HZ, x_spec.t is in seconds
                self.x_spec.t[-1] + (PSG_MAX_IDX - self.x_spec.shape[0]),
                PSG_MAX_IDX)
        elif self.x_spec.shape[0] > PSG_MAX_IDX:
            self.x_spec.Zxx = self.x_spec.Zxx[:PSG_MAX_IDX, :]
            self.x_spec.t = self.x_spec.t[:PSG_MAX_IDX]

        if self.y.shape[0] < PSG_MAX_IDX:
            self.y = np.pad(
                self.y,
                (0, PSG_MAX_IDX - self.y.shape[0]),
                mode='constant',
                constant_values=MASK_VALUE)
        elif self.y.shape[0] > PSG_MAX_IDX:
            self.y = self.y[:PSG_MAX_IDX]


def resample_walch_dataset(walch_set: DataSetObject, resampled_acc_hz: int = 64) -> DataSetObject:
    """Walch et al's data set has accelerometer at 50 Hz, and PSG every 30 seconds.

    This function adjusts that to fit our setup here by doing 2 things:
    - resampling the accelerometer data to 64 Hz
    - 30x repeating the PSG labels, so we have 1 Hz labels like we have preprocessed with DREAMT.
    """
    new_set_name = f'{walch_set.name}_{resampled_acc_hz}hz'
    new_set = DataSetObject(new_set_name, walch_set.path.parent / new_set_name)
    new_set.ids = walch_set.ids
    # new_set.features = ['accelerometer', 'psg']
    id_tqdm = tqdm(walch_set.ids)
    for idno in id_tqdm:
        id_tqdm.set_description(f'Processing {idno}')
        try:
            accel_data = walch_set.get_feature_data('accelerometer', idno).to_numpy()
            accel_data = accel_data[accel_data[:, 0].argsort()]
            psg_data = walch_set.get_feature_data('psg', idno).to_numpy()
            psg_data = psg_data[psg_data[:, 0].argsort()]

            # Get PSG time
            psg_y = psg_data[:, 1]
            psg_t = np.round(psg_data[:, 0])  # convert to pure seconds
            # resample the accelerometer data to 64 Hz
            accel_raw = accel_data[:, 1:]
            accel_raw_t = accel_data[:, 0]
            max_gap = 0.25
            # bool array that's True when it's a gap.
            gap_idx = np.diff(accel_raw_t) > max_gap  # seconds

            for acc_t_idx, acc_t in enumerate(accel_raw_t[:-1]):
                if not gap_idx[acc_t_idx]:
                    continue
                psg_gap_select = (psg_t >= acc_t) & (psg_t < acc_t + max_gap)
                psg_y[psg_gap_select] = MASK_VALUE
            
            # resample PSG to 1 Hz from the given one
            psg_gap = psg_t[1] - psg_t[0]
            repeat_n = int(psg_gap)
            psg_y_1hz = np.repeat(psg_y, repeat_n)
            psg_t_1hz = np.linspace(psg_t[0], psg_t[-1] + psg_gap - 1, len(psg_y_1hz))

            psg_df = pl.DataFrame({
                'TIMESTAMP': psg_t_1hz,
                'PSG': psg_y_1hz})


            t_50 = np.arange(accel_raw_t[0], accel_raw_t[-1], 1/50)

            accel_50 = np.zeros((len(t_50), 3))
            for i in range(3):
                accel_50[:, i] = np.interp(t_50, accel_raw_t, accel_raw[:, i])
            
            accel_64 = signal.resample(
                accel_50,
                int(len(accel_data) * resampled_acc_hz / 50))
            t_64 = np.linspace(
                accel_raw_t[0],
                accel_raw_t[-1] + 1/resampled_acc_hz,
                len(accel_64))
            accel_df = pl.DataFrame({
                'TIMESTAMP': t_64,
                'ACC_X': accel_64[:, 0],
                'ACC_Y': accel_64[:, 1],
                'ACC_Z': accel_64[:, 2]
            })

            new_set.set_feature_data('accelerometer', idno, accel_df)
            new_set.set_feature_data('psg', idno, psg_df)
        except Exception as e:
            print(f"Error processing {idno}: {e}")

    return new_set


if __name__ == '__main__':
    sets = DataSetObject.find_data_sets(Path('data'))
    walch = sets['walch_et_al']
    start_time = time.time()
    walch_resampled = resample_walch_dataset(walch)
    print(f"Resampling took {time.time() - start_time:.2f} seconds.")
    walch_resampled.save_set(walch_resampled.path)
    print("Resampled data saved.")