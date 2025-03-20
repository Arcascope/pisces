

from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from examples.dreamt_acc.constants import MASK_VALUE, PSG_MAX_IDX, TIMESTAMP_HZ


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
            self.specgram = self.specgram[:, self.f >= freq_min]
            self.f = self.f[self.f >= freq_min]
        if freq_max is not None:
            self.specgram = self.specgram[:, self.f <= freq_max]
            self.f = self.f[self.f <= freq_max]
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
                  extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]],
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