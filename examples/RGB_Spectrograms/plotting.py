import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from examples.NHRC.src.make_triplots import get_logreg_data, get_summary_string, plot_single_person
from examples.RGB_Spectrograms.constants import ACC_HZ, N_OUTPUT_EPOCHS, rgb_saved_predictions_name
from examples.RGB_Spectrograms.preprocessing import prepare_data
from examples.RGB_Spectrograms.utils import load_preprocessed_data
from pisces.metrics import apply_threshold, threshold_from_binary_search

plt.rcParams['font.family'] = 'Arial'
COLOR_PALETTE = sns.color_palette("colorblind")


def overlay_channels_fixed(spectrogram_tensor, mintile=5, maxtile=95, ax=None, clip: bool = False):
    """
    Overlay spectrogram channels as an RGB image by stacking the three axes (x, y, z).
    
    Parameters:
        spectrogram_tensor (numpy.ndarray): Spectrogram tensor of shape (time_bins, freq_bins, 3).
    """
    # Normalize each channel to [0, 1] for proper RGB visualization
    norm_spec = np.zeros_like(np.squeeze(spectrogram_tensor))
    if clip:
        for i in range(3):
            channel = spectrogram_tensor[:, :, i]
            p5, p95 = np.percentile(channel, [mintile, maxtile])  # Robust range
            norm_spec[:, :, i] = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)  # Avoid dividing by zero
    else:
        spec_max = np.max(spectrogram_tensor)
        spec_min = np.min(spectrogram_tensor)
        norm_spec = (spectrogram_tensor - spec_min) / (spec_max - spec_min)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # Display the combined RGB image

    ax.imshow(norm_spec, aspect='auto', origin='lower', interpolation='none')
    # add_rgb_legend(plt.gca())
    # plt.colorbar(label='Intensity')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Frequency Bins')
    ax.set_title('Overlayed Spectrogram Channels as RGB')

def debug_plot(predictions, spectrogram_3d, y_true, weights: np.ndarray | None = None, saveto: str = None, wasa_threshold: float | None = None, wasa_value: float | None = None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    overlay_channels_fixed(np.swapaxes(np.squeeze(spectrogram_3d), 0, 1), ax=axs[0], clip=False)
    predictions_squeezed = np.squeeze(predictions)
    print(predictions_squeezed.shape)
    try:
        axs[1].plot(range(N_OUTPUT_EPOCHS), predictions_squeezed)
    except:
        try:
            axs[1].stackplot(range(N_OUTPUT_EPOCHS), predictions_squeezed.T)
        except Exception as e:
            print(e)

    if wasa_threshold is not None:
        axs[1].axhline(wasa_threshold, color='r', linestyle='--', label='WASA Threshold')
        binary_predictions = predictions_squeezed > wasa_threshold
        axs[1].plot(range(N_OUTPUT_EPOCHS), binary_predictions, label='Threshold applied')
    
    if wasa_value is not None:
        axs[1].set_title(f'WASA: {wasa_value:.2f}')

    axs[1].plot(range(N_OUTPUT_EPOCHS), y_true, 'k--')
    axs[1].set_xlim([0, N_OUTPUT_EPOCHS])
    axs[1].legend()
    # axs[1].set_ylim([0, 1])
    if weights is not None:
        # apply gray vertical bar over any regions with weight 0.0
        for idx in np.where(weights == 0.0)[0]:
            axs[1].axvspan(idx, idx+1, color='gray', alpha=0.5)
    fig.tight_layout(pad=0.1)
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto)
    plt.close()
    print("debug plot complete")


def create_histogram_rgb(run_mode: str, preprocessed_data_path: Path, saved_output_dir: Path, acc_hz: int = ACC_HZ, TARGET_SLEEP: float = 0.95, sleep_proba: bool = True):
    start_run = time.time()

    # Load stationary data
    static_preprocessed_data = load_preprocessed_data("stationary"
, preprocessed_data_path)

    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data)

    # Load hybrid data
    hybrid_preprocessed_data = load_preprocessed_data("hybrid", preprocessed_data_path)
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    # Holders for outputs
    static_performs = []
    hybrid_static_thresh_performs = []
    hybrid_best_thresh_performs = []

    for i, key in enumerate(static_keys):
        print(f"Comparing {key}")
        static_predictions: np.ndarray
        hybrid_predictions: np.ndarray

        if run_mode == "lr":
            static_predictions, hybrid_predictions = get_logreg_data(
                static_data_bundle, hybrid_data_bundle, i)

        if run_mode == "rgb":
            static_predictions = np.squeeze(
                np.load(rgb_saved_predictions_name(key, saved_output_dir=saved_output_dir, set_name="static")))
            hybrid_predictions = np.squeeze(
                np.load(rgb_saved_predictions_name(key, saved_output_dir=saved_output_dir, set_name="hybrid")))
            # the methods below want wake probabilities
            if sleep_proba:
                static_predictions = 1 - static_predictions
                hybrid_predictions = 1 - hybrid_predictions
            # static_predictions = 1 - np.squeeze(
            #     np.load(rgb_saved_predictions_name(key, saved_output_dir=saved_output_dir, set_name="static")))
            # hybrid_predictions = 1 - np.squeeze(
            #     np.load(rgb_saved_predictions_name(key, saved_output_dir=saved_output_dir, set_name="hybrid")))

        true_labels = static_data_bundle.labels[i, :]

        true_labels[true_labels > 1] = 1
        target_sleep_accuracy = TARGET_SLEEP

        static_threshold = threshold_from_binary_search(
            true_labels, static_predictions, target_sleep_accuracy)

        print(f"Threshold: {static_threshold}")
        static_perform = apply_threshold(
            true_labels, static_predictions, static_threshold)


        hybrid_static_thresh_perform = apply_threshold(
            true_labels, hybrid_predictions, static_threshold)

        hybrid_threshold = threshold_from_binary_search(
            true_labels, hybrid_predictions, target_sleep_accuracy)

        print("Hybrid Threshold: ", hybrid_threshold)
        hybrid_best_thresh_perform = apply_threshold(
            true_labels, hybrid_predictions, hybrid_threshold)

        summary_string = get_summary_string(static_perform=static_perform,
                                            hybrid_static_thresh_perform=hybrid_static_thresh_perform,
                                            hybrid_best_thresh_perform=hybrid_best_thresh_perform,
                                            name=static_keys[i])

        plot_single_person(static_predictions,
                           hybrid_predictions,
                           true_labels=true_labels,
                           static_threshold=static_threshold,
                           hybrid_threshold=hybrid_threshold,
                           name=static_keys[i],
                           target_sleep_accuracy=target_sleep_accuracy,
                           run_mode=run_mode,
                           title=summary_string)

        # Append values to arrays
        static_performs.append(static_perform)
        hybrid_static_thresh_performs.append(hybrid_static_thresh_perform)
        hybrid_best_thresh_performs.append(hybrid_best_thresh_perform)

        plt.close()

    metric_colors = {
        'sleep_accuracy': COLOR_PALETTE[4], 'tst_error': COLOR_PALETTE[1], 'wake_accuracy': COLOR_PALETTE[2]}

    # After the loop, create histograms
    fig, axs = plt.subplots(3, 3, figsize=(9, 7))

    static_sleep_accuracies = [x.sleep_accuracy for x in static_performs]
    static_wake_accuracies = [x.wake_accuracy for x in static_performs]
    static_tst_errors = [x.tst_error for x in static_performs]

    sasa_linspace = np.linspace(0, 1, 27)
    axs[0, 0].hist(static_sleep_accuracies, bins=sasa_linspace,
                   color=metric_colors['sleep_accuracy'], 
                   alpha=0.7,)
                #    edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[0, 1].hist(static_wake_accuracies, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'],
                   alpha=0.7,)
                #    edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[0, 2].hist(static_tst_errors, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'],
                   alpha=0.7,)
                #    edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    hybrid_sleep_accuracies_static_thresh = [
        x.sleep_accuracy for x in hybrid_static_thresh_performs]
    hybrid_wake_accuracies_static_thresh = [
        x.wake_accuracy for x in hybrid_static_thresh_performs]
    hybrid_tst_errors_static_thresh = [
        x.tst_error for x in hybrid_static_thresh_performs]

    axs[1, 0].hist(hybrid_sleep_accuracies_static_thresh, bins=sasa_linspace,
                   color=metric_colors['sleep_accuracy'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[1, 1].hist(hybrid_wake_accuracies_static_thresh, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[1, 2].hist(hybrid_tst_errors_static_thresh, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    hybrid_sleep_choose_best = [
        x.sleep_accuracy for x in hybrid_best_thresh_performs]
    hybrid_wake_choose_best = [
        x.wake_accuracy for x in hybrid_best_thresh_performs]
    hybrid_tst_choose_best = [
        x.tst_error for x in hybrid_best_thresh_performs]

    axs[2, 0].hist(hybrid_sleep_choose_best, bins=sasa_linspace,
                   color=metric_colors['sleep_accuracy'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[2, 1].hist(hybrid_wake_choose_best, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[2, 2].hist(hybrid_tst_choose_best, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,)
                #    edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    # Set common properties for all axes
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('')
        ax.set_ylim(0, 32 if ax in axs[:, 0] else 6 if ax in axs[:, 1] else 10)
        ax.set_xlim(-50 if ax in axs[:, 2] else 0,
                    50 if ax in axs[:, 2] else 1)

    # Add row labels to the left of the leftmost plots
    row_labels = ["Static data\nevaluated with\nstatic threshold",
                  "Hybrid data\nevaluated with\nstatic threshold",
                  "Hybrid data\nevaluated with\nhybrid threshold"]
    for ax, label in zip(axs[:, 0], row_labels):
        ax.set_ylabel(label, rotation=0, size='x-large',
                      labelpad=80, ha='center')
    # Uncomment to change y-label to "Count" instead of the experiment conditions
    # for ax, label in zip(axs[:, 0], ['Count'] * 3):
    #     ax.set_ylabel(label)
    for ax, label in zip(axs[2, :], ['Sleep accuracy', 'Wake accuracy', 'TST error (minutes)']):
        ax.set_xlabel(label)

    for i, data in enumerate([static_sleep_accuracies, static_wake_accuracies, static_tst_errors,
                              hybrid_sleep_accuracies_static_thresh, hybrid_wake_accuracies_static_thresh, hybrid_tst_errors_static_thresh,
                              hybrid_sleep_choose_best, hybrid_wake_choose_best, hybrid_tst_choose_best]):
        row = i // 3
        col = i % 3
        mean_value = np.mean(data)
        axs[row, col].axvline(mean_value, color='red',
                              linestyle='dashed', linewidth=1)
        abs_mean_value = np.mean(np.abs(np.array(data)))

        # Add text showing the mean_value above the line
        if mean_value == abs_mean_value:
            axs[row, col].text(mean_value, axs[row, col].get_ylim()[
                1], f'Mean: {mean_value:.2f}', color='red', ha='center', fontsize=8)
        else:
            axs[row, col].text(mean_value, axs[row, col].get_ylim()[
                1], f'Mean: {mean_value:.2f} (Abs. Mean: {abs_mean_value:.2f})', color='red', ha='center', fontsize=8)
        if col == 2:
            percentage_above = np.sum(np.abs(data) > 30) / len(data) * 100
            axs[row, col].text(
                -20, axs[row, col].get_ylim()[1] - 2, f'% >30 min:\n{percentage_above:.2f}%', color='gray', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(
        saved_output_dir / f"{int(time.time())}_{run_mode}_{acc_hz}_hists_WASA{int(target_sleep_accuracy * 100)}.png", dpi=200)
    # plt.show()
    plt.close()

    # Do a statistical test to determine if the distributions are different
    from scipy.stats import ttest_ind
    static_sleep_accuracies = np.array(static_sleep_accuracies)
    hybrid_sleep_accuracies_static_thresh = np.array(
        hybrid_sleep_accuracies_static_thresh)
    hybrid_sleep_choose_best = np.array(hybrid_sleep_choose_best)

    print("Statistical test result:")
    print(ttest_ind(static_sleep_accuracies, hybrid_sleep_accuracies_static_thresh))

    end_run = time.time()
    print(f"Total time to make triplots: {end_run - start_run} seconds")
    print("Done with triplots")
