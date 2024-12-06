import os
import matplotlib.pyplot as plt
import numpy as np

from examples.RGB_Spectrograms.constants import N_OUTPUT_EPOCHS


def overlay_channels_fixed(spectrogram_tensor, mintile=5, maxtile=95, ax=None):
    """
    Overlay spectrogram channels as an RGB image by stacking the three axes (x, y, z).
    
    Parameters:
        spectrogram_tensor (numpy.ndarray): Spectrogram tensor of shape (time_bins, freq_bins, 3).
    """
    # Normalize each channel to [0, 1] for proper RGB visualization
    norm_spec = np.zeros_like(spectrogram_tensor)
    for i in range(3):
        channel = spectrogram_tensor[:, :, i]
        p5, p95 = np.percentile(channel, [mintile, maxtile])  # Robust range
        norm_spec[:, :, i] = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)  # Avoid dividing by zero
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Display the combined RGB image

    ax.imshow(np.squeeze(norm_spec), aspect='auto', origin='lower')
    # add_rgb_legend(plt.gca())
    # plt.colorbar(label='Intensity')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Frequency Bins')
    ax.set_title('Overlayed Spectrogram Channels as RGB')

def debug_plot(predictions, spectrogram_3d, weights: np.ndarray | None = None, saveto: str = None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    overlay_channels_fixed(np.swapaxes(spectrogram_3d, 0, 1), ax=axs[0])
    axs[1].stackplot(range(N_OUTPUT_EPOCHS), predictions.T)
    axs[1].set_xlim([0, N_OUTPUT_EPOCHS])
    axs[1].set_ylim([0, 1])
    if weights is not None:
        # apply gray vertical bar over any regions with weight 0.0
        for idx in np.where(weights == 0.0)[0]:
            axs[1].axvspan(idx, idx+1, color='gray', alpha=0.5)
    fig.tight_layout(pad=0.1)
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto)
    plt.close()