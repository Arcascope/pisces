import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from analyses.NHRC.nhrc_utils.analysis import prepare_data
from constants import ACC_HZ as acc_hz, TARGET_SLEEP

plt.rcParams['font.family'] = 'Arial'
COLOR_PALETTE = sns.color_palette("colorblind")


class PerformanceMetrics:
    def __init__(self, sleep_accuracy, wake_accuracy, tst_error):
        self.sleep_accuracy = sleep_accuracy
        self.wake_accuracy = wake_accuracy
        self.tst_error = tst_error


def apply_threshold(labels, predictions, threshold):
    true_wakes = np.where(labels == 0)[0]
    predicted_wakes = np.where(predictions > threshold)[0]

    # calculate the number of true positives
    wake_accuracy = len(set(true_wakes).intersection(
        set(predicted_wakes))) / len(true_wakes)

    # calculate the sleep accuracy
    true_sleeps = np.where(labels > 0)[0]
    predicted_sleeps = np.where((predictions <= threshold) & (labels != -1))[0]

    sleep_accuracy = len(set(true_sleeps).intersection(
        set(predicted_sleeps))) / len(true_sleeps)

    tst_error = (len(true_sleeps) - len(predicted_sleeps)) / 2  # Minutes

    return PerformanceMetrics(sleep_accuracy, wake_accuracy, tst_error)


def threshold_from_binary_search(labels, wake_probabilities,
                                 target_sleep_accuracy) -> float:

    # How close to the target wake false positive rate we need to be before stopping
    false_positive_buffer = 0.0001
    fraction_sleep_scored_as_sleep = -1
    binary_search_counter = 0

    max_attempts_binary_search = 50

    # While we haven't found the target wake false positive rate
    # (and haven't exceeded the number of allowable searches), keep searching:
    while (
        fraction_sleep_scored_as_sleep < target_sleep_accuracy - false_positive_buffer
        or fraction_sleep_scored_as_sleep
        >= target_sleep_accuracy + false_positive_buffer
    ) and binary_search_counter < max_attempts_binary_search:
        # If this is the first iteration on the binary search, initialize.
        if binary_search_counter == 0:
            threshold_for_sleep = 0.5
            threshold_delta = 0.25
        else:
            if (
                fraction_sleep_scored_as_sleep
                < target_sleep_accuracy - false_positive_buffer
            ):
                threshold_for_sleep = threshold_for_sleep + threshold_delta
                threshold_delta = threshold_delta / 2

            if (
                fraction_sleep_scored_as_sleep
                >= target_sleep_accuracy + false_positive_buffer
            ):
                threshold_for_sleep = threshold_for_sleep - threshold_delta
                threshold_delta = threshold_delta / 2

        performance = apply_threshold(
            labels, wake_probabilities, threshold_for_sleep)
        fraction_sleep_scored_as_sleep = performance.sleep_accuracy
        print("Fraction sleep correct: " + str(fraction_sleep_scored_as_sleep))
        print("Goal fraction sleep correct: " + str(target_sleep_accuracy))
        binary_search_counter = binary_search_counter + 1

    print("Declaring victory with " +
          str(fraction_sleep_scored_as_sleep) + "\n\n")

    print("Goal was: " + str(target_sleep_accuracy))
    return threshold_for_sleep


def plot_single_person(static_predictions,
                       hybrid_predictions,
                       true_labels,
                       static_threshold,
                       hybrid_threshold,
                       name,
                       target_sleep_accuracy,
                       run_mode, title):

    # Make a 2x1 plot
    fig, axs = plt.subplots(2, 1)

    x = np.arange(len(static_predictions))

    masked_indices = true_labels == -1
    static_predictions[masked_indices] = np.nan
    hybrid_predictions[masked_indices] = np.nan

    max_index = len(true_labels)

    axs[0].plot(x[:max_index],
                static_predictions[:max_index], color=COLOR_PALETTE[3], label="Static")
    axs[0].plot(x[:max_index],
                hybrid_predictions[:max_index], color=COLOR_PALETTE[6], label="Hybrid")
    axs[0].plot(x[:max_index], np.ones_like(x[:max_index]) * static_threshold,
                '--', color=COLOR_PALETTE[2],
                label=f"Threshold for\n{int(100*target_sleep_accuracy)}% sleep accuracy\nin static data")
    axs[0].plot(x[:max_index], np.ones_like(x[:max_index]) * hybrid_threshold,
                '--', color=COLOR_PALETTE[7],
                label=f"Threshold for\n{int(100*target_sleep_accuracy)}% sleep accuracy\nin hybrid data")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Wake probability")
    axs[0].legend(loc='center left', bbox_to_anchor=(
        1, 0.5), fontsize=8, frameon=False)

    static_predicted_labels = np.ones_like(static_predictions)
    static_predicted_labels[static_predictions > static_threshold] = 0
    static_predicted_labels[true_labels == -1] = np.nan

    hybrid_labels_using_static_threshold = np.ones_like(hybrid_predictions)
    hybrid_labels_using_static_threshold[hybrid_predictions >
                                         static_threshold] = 0
    hybrid_labels_using_static_threshold[true_labels == -1] = np.nan

    hybrid_labels = np.ones_like(hybrid_predictions)
    hybrid_labels[hybrid_predictions > hybrid_threshold] = 0
    hybrid_labels[true_labels == -1] = np.nan

    true_labels[true_labels == -1] = np.nan

    x = x[:max_index]
    axs[1].step(x, true_labels + 6, 'k', label="True labels")
    axs[1].step(x, static_predicted_labels + 4,
                color=COLOR_PALETTE[2], label="Static predictions")
    axs[1].step(x, hybrid_labels_using_static_threshold +
                2, color=COLOR_PALETTE[9], label="Hybrid predictions\nusing static\nthreshold")
    axs[1].step(x, hybrid_labels,
                color=COLOR_PALETTE[7],
                label="Hybrid predictions\nusing tuned\nthreshold")
    axs[1].set_xlabel("Epoch")
    axs[1].spines['left'].set_visible(False)

    # Hide y-axis for axs[1]
    axs[1].set_yticks([])

    # Turn off the top and right spines
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Move legend to the right of the plot and make font small
    axs[1].legend(loc='center left', bbox_to_anchor=(
        1, 0.5), fontsize=8, frameon=False)
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"debug/{run_mode}_{name}.png", dpi=200)


def get_logreg_data(static_data_bundle, hybrid_data_bundle, i):
    static_activity_data = static_data_bundle.activity[i, :].numpy()
    hybrid_activity_data = hybrid_data_bundle.activity[i, :].numpy()

    static_activity_data[static_activity_data < 0] = 0
    hybrid_activity_data[hybrid_activity_data < 0] = 0

    # Convolve with a blur kernel of length 41
    kernel_width = 61
    start_ind = int(kernel_width / 2)
    static_predictions = np.convolve(
        static_activity_data, np.exp(-np.linspace(-2, 2, kernel_width)**2), mode='same')
    hybrid_predictions = np.convolve(
        hybrid_activity_data, np.exp(-np.linspace(-2, 2, kernel_width)**2), mode='same')

    # Take every other element
    static_predictions = static_predictions[start_ind:-start_ind:2]
    hybrid_predictions = hybrid_predictions[start_ind:-start_ind:2]

    static_predictions = static_predictions / \
        np.max(static_predictions)
    hybrid_predictions = hybrid_predictions / \
        np.max(hybrid_predictions)

    return static_predictions, hybrid_predictions
    # One way of doing it, but producing strange results:
    # static_predictions = np.squeeze(np.load(f"{key}_logreg_pred.npy"))
    # hybrid_predictions = np.squeeze(np.load(f"{key}_logreg_pred.npy"))


def get_summary_string(static_perform, hybrid_static_thresh_perform, hybrid_best_thresh_perform, name):
    static_performance_string = f"Stationary Sleep Accuracy: {static_perform.sleep_accuracy:.2f}, Wake Accuracy: {static_perform.wake_accuracy:.2f}, TST Error: {static_perform.tst_error} min"
    hybrid_performance_string = f"Hybrid Sleep Accuracy w/ Stationary Threshold: {hybrid_static_thresh_perform.sleep_accuracy:.2f}, Wake Accuracy: {hybrid_static_thresh_perform.wake_accuracy:.2f}, TST Error: {hybrid_static_thresh_perform.tst_error} min"
    hybrid_performance_string_best_thresh = f"Hybrid Sleep Accuracy: {hybrid_best_thresh_perform.sleep_accuracy:.2f}, Wake Accuracy: {hybrid_best_thresh_perform.wake_accuracy:.2f}, TST Error: {hybrid_best_thresh_perform.tst_error} min"
    return f"{name}\n{static_performance_string}\n{hybrid_performance_string}\n{hybrid_performance_string_best_thresh}"


def create_histogram(run_mode="naive"):
    start_run = time.time()

    # Load stationary data
    dataset = "stationary"
    static_preprocessed_data = np.load(f'./pre_processed_data/{dataset}/{dataset}_preprocessed_data_{acc_hz}.npy',
                                       allow_pickle=True).item()

    static_keys = list(static_preprocessed_data.keys())
    static_data_bundle = prepare_data(static_preprocessed_data)

    # Load hybrid data
    dataset = "hybrid"
    hybrid_preprocessed_data = np.load(f'./pre_processed_data/{dataset}/{dataset}_preprocessed_data_{acc_hz}.npy',
                                       allow_pickle=True).item()
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    # Holders for outputs
    static_performs = []
    hybrid_static_thresh_performs = []
    hybrid_best_thresh_performs = []

    for i, key in enumerate(static_keys):
        print(f"Comparing {key}")

        if run_mode == "lr":
            static_predictions, hybrid_predictions = get_logreg_data(
                static_data_bundle, hybrid_data_bundle, i)

        if run_mode == "finetune":
            static_predictions = 1 - np.squeeze(
                np.load(f"{key}_cnn_pred_static.npy"))
            hybrid_predictions = 1 - np.squeeze(
                np.load(f"{key}_cnn_pred_hybrid.npy"))

        if run_mode == "naive":
            static_predictions = static_data_bundle.mo_predictions[i, :, 0].numpy(
            )
            hybrid_predictions = hybrid_data_bundle.mo_predictions[i, :, 0].numpy(
            )

        true_labels = static_data_bundle.true_labels[i, :].numpy()

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
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))

    static_sleep_accuracies = [x.sleep_accuracy for x in static_performs]
    static_wake_accuracies = [x.wake_accuracy for x in static_performs]
    static_tst_errors = [x.tst_error for x in static_performs]

    axs[0, 0].hist(static_sleep_accuracies, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[0, 1].hist(static_wake_accuracies, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[0, 2].hist(static_tst_errors, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    hybrid_sleep_accuracies_static_thresh = [
        x.sleep_accuracy for x in hybrid_static_thresh_performs]
    hybrid_wake_accuracies_static_thresh = [
        x.wake_accuracy for x in hybrid_static_thresh_performs]
    hybrid_tst_errors_static_thresh = [
        x.tst_error for x in hybrid_static_thresh_performs]

    axs[1, 0].hist(hybrid_sleep_accuracies_static_thresh, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[1, 1].hist(hybrid_wake_accuracies_static_thresh, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[1, 2].hist(hybrid_tst_errors_static_thresh, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    hybrid_sleep_choose_best = [
        x.sleep_accuracy for x in hybrid_best_thresh_performs]
    hybrid_wake_choose_best = [
        x.wake_accuracy for x in hybrid_best_thresh_performs]
    hybrid_tst_choose_best = [
        x.tst_error for x in hybrid_best_thresh_performs]

    axs[2, 0].hist(hybrid_sleep_choose_best, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[2, 1].hist(hybrid_wake_choose_best, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[2, 2].hist(hybrid_tst_choose_best, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    # Set common properties for all axes
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('')
        ax.set_ylim(0, 32 if ax in axs[:, 0] else 6 if ax in axs[:, 1] else 10)
        ax.set_xlim(-50 if ax in axs[:, 2] else 0,
                    50 if ax in axs[:, 2] else 1)

    # Set specific labels
    for ax, label in zip(axs[:, 0], ['Count'] * 3):
        ax.set_ylabel(label)
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
        axs[row, col].text(mean_value, axs[row, col].get_ylim()[
                           1], f'Mean: {mean_value:.2f}', color='red', ha='center', fontsize=8)
        if col == 2:
            percentage_above = np.sum(np.abs(data) > 30) / len(data) * 100
            axs[row, col].text(
                -20, axs[row, col].get_ylim()[1] - 2, f'% >30 min:\n{percentage_above:.2f}%', color='gray', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{run_mode}_hists_WASA{int(target_sleep_accuracy * 100)}.png")
    plt.show()

    end_run = time.time()
    print(f"Total time to make triplots: {end_run - start_run} seconds")
    print("Done with triplots")


if __name__ == "__main__":
    # run_mode = "finetune"  # "naive" "lr" "finetune"
    create_histogram("lr")
