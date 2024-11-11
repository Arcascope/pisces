from analyses.NHRC.nhrc_utils.analysis import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
plt.rcParams['font.family'] = 'Arial'
COLOR_PALETTE = sns.color_palette("colorblind")


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

    return sleep_accuracy, wake_accuracy, tst_error


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

        fraction_sleep_scored_as_sleep, _, _ = apply_threshold(
            labels, wake_probabilities, threshold_for_sleep)
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
                       target_sleep_accuracy):

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
                '--', color=COLOR_PALETTE[5],
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
                color=COLOR_PALETTE[5],
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

    # Include hybrid and static sleep and wake accuracies in the title
    static_performance_string = f"Stationary Sleep Accuracy: {static_sleep_accuracy:.2f}, Wake Accuracy: {static_wake_accuracy:.2f}, TST Error: {static_tst_error} min"
    hybrid_performance_string = f"Hybrid Sleep Accuracy w/ Stationary Threshold: {hybrid_sleep_accuracy_static_thresh:.2f}, Wake Accuracy: {hybrid_wake_accuracy_static_thresh:.2f}, TST Error: {hybrid_tst_error_static_thresh} min"
    hybrid_performance_string_best_thresh = f"Hybrid Sleep Accuracy: {hybrid_sleep_accuracy_choose_best_threshold:.2f}, Wake Accuracy: {hybrid_wake_accuracy_choose_best_threshold:.2f}, TST Error: {hybrid_tst_error_choose_best_threshold} min"
    plt.suptitle(
        f"{name}\n{static_performance_string}\n{hybrid_performance_string}\n{hybrid_performance_string_best_thresh}", fontsize=8)

    plt.tight_layout()
    plt.savefig(name + ".png", dpi=200)


if __name__ == "__main__":

    start_run = time.time()
    acc_hz = "50"

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
    hybrid_keys = list(hybrid_preprocessed_data.keys())
    hybrid_data_bundle = prepare_data(hybrid_preprocessed_data)

    # Holders for outputs
    static_sleep_accuracies = []
    static_wake_accuracies = []
    static_tst_errors = []

    hybrid_sleep_accuracies_static_thresh = []
    hybrid_wake_accuracies_static_thresh = []
    hybrid_tst_errors_static_thresh = []

    hybrid_sleep_choose_best = []
    hybrid_wake_choose_best = []
    hybrid_tst_choose_best = []

    for i, key in enumerate(static_keys):
        print(f"Comparing {key}")
        static_predictions = static_data_bundle.mo_predictions[i, :, 0].numpy()
        hybrid_predictions = hybrid_data_bundle.mo_predictions[i, :, 0].numpy()
        true_labels = static_data_bundle.true_labels[i, :].numpy()

        true_labels[true_labels > 1] = 1
        unmasked_indices = true_labels != -1
        target_sleep_accuracy = 0.93

        static_threshold = threshold_from_binary_search(
            true_labels, static_predictions, target_sleep_accuracy)

        print(f"Threshold: {static_threshold}")
        static_sleep_accuracy, static_wake_accuracy, static_tst_error = apply_threshold(
            true_labels, static_predictions, static_threshold)

        hybrid_sleep_accuracy_static_thresh, hybrid_wake_accuracy_static_thresh, hybrid_tst_error_static_thresh = apply_threshold(
            true_labels, hybrid_predictions, static_threshold)

        hybrid_threshold = threshold_from_binary_search(
            true_labels, hybrid_predictions, target_sleep_accuracy)

        print("Hybrid Threshold: ", hybrid_threshold)
        hybrid_sleep_accuracy_choose_best_threshold, hybrid_wake_accuracy_choose_best_threshold, hybrid_tst_error_choose_best_threshold = apply_threshold(
            true_labels, hybrid_predictions, hybrid_threshold)

        plot_single_person(static_predictions,
                           hybrid_predictions,
                           true_labels=true_labels,
                           static_threshold=static_threshold,
                           hybrid_threshold=hybrid_threshold,
                           name=static_keys[i],
                           target_sleep_accuracy=target_sleep_accuracy)

        # Append values to arrays
        static_sleep_accuracies.append(static_sleep_accuracy)
        static_wake_accuracies.append(static_wake_accuracy)
        static_tst_errors.append(static_tst_error)

        hybrid_sleep_accuracies_static_thresh.append(
            hybrid_sleep_accuracy_static_thresh)
        hybrid_wake_accuracies_static_thresh.append(
            hybrid_wake_accuracy_static_thresh)
        hybrid_tst_errors_static_thresh.append(hybrid_tst_error_static_thresh)

        hybrid_sleep_choose_best.append(
            hybrid_sleep_accuracy_choose_best_threshold)
        hybrid_wake_choose_best.append(
            hybrid_wake_accuracy_choose_best_threshold)
        hybrid_tst_choose_best.append(
            hybrid_tst_error_choose_best_threshold)

        plt.close()

    metric_colors = {
        'sleep_accuracy': COLOR_PALETTE[4], 'tst_error': COLOR_PALETTE[1], 'wake_accuracy': COLOR_PALETTE[2]}

    # After the loop, create histograms
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))

    axs[0, 0].hist(static_sleep_accuracies, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[0, 1].hist(static_wake_accuracies, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[0, 2].hist(static_tst_errors, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)
    axs[1, 0].hist(hybrid_sleep_accuracies_static_thresh, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[1, 1].hist(hybrid_wake_accuracies_static_thresh, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[1, 2].hist(hybrid_tst_errors_static_thresh, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)
    axs[2, 0].hist(hybrid_sleep_choose_best, bins=np.linspace(0, 1, 21),
                   color=metric_colors['sleep_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['sleep_accuracy']) * 0.8)
    axs[2, 1].hist(hybrid_wake_choose_best, bins=np.linspace(0, 1, 21),
                   color=metric_colors['wake_accuracy'], alpha=0.7,
                   edgecolor=np.array(metric_colors['wake_accuracy']) * 0.8)
    axs[2, 2].hist(hybrid_tst_choose_best, bins=np.linspace(-50, 50, 21),
                   color=metric_colors['tst_error'], alpha=0.7,
                   edgecolor=np.array(metric_colors['tst_error']) * 0.8)

    for ax in axs.flat:
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('')

    axs[0, 2].set_xlim(-50, 50)
    axs[1, 2].set_xlim(-50, 50)
    axs[2, 2].set_xlim(-50, 50)
    axs[0, 0].set_ylim(0, 32)
    axs[1, 0].set_ylim(0, 32)
    axs[2, 0].set_ylim(0, 32)
    axs[0, 1].set_ylim(0, 6)
    axs[1, 1].set_ylim(0, 6)
    axs[2, 1].set_ylim(0, 6)
    axs[0, 0].set_ylabel('Count')
    axs[1, 0].set_ylabel('Count')
    axs[2, 0].set_ylabel('Count')
    axs[2, 0].set_xlabel('Sleep accuracy')
    axs[2, 1].set_xlabel('Wake accuracy')
    axs[2, 2].set_xlabel('TST error (minutes)')

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
    plt.savefig("hists.png")
    plt.show()

    end_run = time.time()
    print(f"Total time: {end_run - start_run}")
    print("Done with everything")
