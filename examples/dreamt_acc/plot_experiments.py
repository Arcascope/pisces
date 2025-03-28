from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from conv2d_net import TrainingResult

def plot(experiment_results_csv: Path):
    """"
    Plot the experiment results from a CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(experiment_results_csv)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12))

    plt.rcParams.update({'font.size': 14})

    ID_COL = TrainingResult.id_column()
    EXPERIMENT_COL = TrainingResult.experiment_id_column()
    WAKE_ACC_COL = TrainingResult.wake_acc_column()
    EPOCH_TIME_COL = TrainingResult.epoch_seconds_column()
    SLEEP_ACC_COL = TrainingResult.sleep_acc_column()

    # Get the experiment plot axis (top-left)
    experiment_plot_axis = axes[0, 0]

    wake_df = df[[ID_COL, EXPERIMENT_COL, EPOCH_TIME_COL, WAKE_ACC_COL]].copy()
    # sort wake_df by epoch time
    wake_df = wake_df.sort_values(by=EPOCH_TIME_COL)
    # drop rows with NaN values
    wake_df = wake_df.dropna()

    # Get unique experiment IDs ordered by time
    exp_time_df = wake_df[[EXPERIMENT_COL, EPOCH_TIME_COL]].drop_duplicates()
    exp_time_df = exp_time_df.sort_values(by=EPOCH_TIME_COL)
    ordered_exp_ids = exp_time_df[EXPERIMENT_COL].values
    last_experiment_hash = ordered_exp_ids[-1]

    # Set x-axis ordering based on experiment time
    experiment_plot_axis.set_xticks(range(len(ordered_exp_ids)))
    experiment_plot_axis.set_xticklabels([])

    # Group by test_id and plot
    for test_id, group in wake_df.groupby(ID_COL):
        # Map experiment hash to its position in time order
        x_pos = [np.where(ordered_exp_ids == exp)[0][0] for exp in group[EXPERIMENT_COL]]
        experiment_plot_axis.plot(x_pos, group[WAKE_ACC_COL], alpha=0.5)

    # Calculate median and mean while preserving order
    median_values = []
    mean_values = []
    x_positions = []

    for i, exp_id in enumerate(ordered_exp_ids):
        exp_data = wake_df[wake_df[EXPERIMENT_COL] == exp_id]
        if not exp_data.empty:
            median_values.append(exp_data[WAKE_ACC_COL].median())
            mean_values.append(exp_data[WAKE_ACC_COL].mean())
            x_positions.append(i)

    # Plot median and mean in the correct order
    experiment_plot_axis.plot(x_positions, median_values, label='Median', color='black', linewidth=4, linestyle='--')
    experiment_plot_axis.fill_between(x_positions, median_values, color='gray', alpha=0.3)
    experiment_plot_axis.plot(x_positions, mean_values, 'x-', label='Mean', color='black', linewidth=2)

    # Now plot the established values "to beat"
    blur_wasa = 0.59
    mo_wasa = 0.66

    experiment_plot_axis.axhline(y=blur_wasa, color='r', linestyle=':', linewidth=4, label='Blur')
    experiment_plot_axis.axhline(y=mo_wasa, color='g', linestyle=':', linewidth=4, label='MO')

    experiment_plot_axis.set_xlabel('Experiment ID (Time Ordered)')
    experiment_plot_axis.set_ylabel('wasa95')
    experiment_plot_axis.set_title('wasa95 grouped by test_id')
    experiment_plot_axis.legend()
    experiment_plot_axis.set_ylim(0.0, 1.0)
    experiment_plot_axis.grid(visible=True, axis='both')

    # Sleep accuracy histogram [0, 1]
    if SLEEP_ACC_COL in df.columns:
        ax = axes[0, 1]
        sleep_data = df[[EXPERIMENT_COL, SLEEP_ACC_COL]].dropna()
        
        # Plot all experiments except the last one in blue with 0.5 opacity
        for i, exp_id in enumerate(ordered_exp_ids[:-1]):
            exp_data = sleep_data[sleep_data[EXPERIMENT_COL] == exp_id]
            if not exp_data.empty:
                ax.hist(exp_data[SLEEP_ACC_COL], bins=20, alpha=0.5, color='blue', label=f'Exp {exp_id}' if i == 0 else None)
        
        # Plot the last experiment in orange
        last_exp_data = sleep_data[sleep_data[EXPERIMENT_COL] == last_experiment_hash]
        if not last_exp_data.empty:
            ax.hist(last_exp_data[SLEEP_ACC_COL], bins=20, alpha=0.7, color='orange', label=f'Latest Exp {last_experiment_hash}')
        
        ax.set_title('Sleep Accuracy Distribution by Experiment')
        ax.set_xlabel('Sleep Accuracy')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)

    # Wake accuracy histogram [0, 2]
    ax = axes[0, 2]
    wake_data = df[[EXPERIMENT_COL, WAKE_ACC_COL]].dropna()

    # Plot all experiments except the last one in blue with 0.5 opacity
    for i, exp_id in enumerate(ordered_exp_ids[:-1]):
        exp_data = wake_data[wake_data[EXPERIMENT_COL] == exp_id]
        if not exp_data.empty:
            ax.hist(exp_data[WAKE_ACC_COL], bins=20, alpha=0.5, color='blue', label=f'Exp {exp_id}' if i == 0 else None)

    # Plot the last experiment in orange
    last_exp_data = wake_data[wake_data[EXPERIMENT_COL] == last_experiment_hash]
    if not last_exp_data.empty:
        ax.hist(last_exp_data[WAKE_ACC_COL], bins=20, alpha=0.7, color='orange', label=f'Latest Exp {last_experiment_hash}')

    ax.set_title('Wake Accuracy Distribution by Experiment')
    ax.set_xlabel('Wake Accuracy')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

    # Create dataframes for all previous experiments and the latest experiment
    previous_df = df[df[EXPERIMENT_COL] != last_experiment_hash]
    last_exp_df = df[df[EXPERIMENT_COL] == last_experiment_hash]

    # Wake accuracy vs spec_max [1, 0]
    if 'max_X' in df.columns:
        # Plot all previous experiments in blue with 0.5 opacity
        sns.scatterplot(x='max_X', y=WAKE_ACC_COL, data=previous_df, ax=axes[1, 0], 
                        alpha=0.5, color='blue', label='Previous Experiments')
        
        # Plot the latest experiment in orange
        sns.scatterplot(x='max_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 0], 
                        alpha=1.0, color='orange', marker='X', label=f'Latest Exp {last_experiment_hash}')
        
        # Only run regplot on the latest experiment data
        if not last_exp_df.empty:
            sns.regplot(x='max_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 0], 
                        scatter=False, color='red')
        
        axes[1, 0].set_title('Wake Accuracy vs Spectrogram Max')
        axes[1, 0].set_xlabel('Spectrogram Max Value')
        axes[1, 0].set_ylabel('Wake Accuracy')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

    # Wake accuracy vs spec_mean [1, 1]
    if 'mean_X' in df.columns:
        # Plot all previous experiments in blue with 0.5 opacity
        sns.scatterplot(x='mean_X', y=WAKE_ACC_COL, data=previous_df, ax=axes[1, 1], 
                        alpha=0.5, color='blue', label='Previous Experiments')
        
        # Plot the latest experiment in orange
        sns.scatterplot(x='mean_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 1], 
                        alpha=1.0, color='orange', marker='X', label=f'Latest Exp {last_experiment_hash}')
        
        # Only run regplot on the latest experiment data
        if not last_exp_df.empty:
            sns.regplot(x='mean_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 1], 
                    scatter=False, color='red')
        
        axes[1, 1].set_title('Wake Accuracy vs Spectrogram Mean')
        axes[1, 1].set_xlabel('Spectrogram Mean Value')
        axes[1, 1].set_ylabel('Wake Accuracy')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

    # Wake accuracy vs spec_std [1, 2]
    if 'std_X' in df.columns:
        # Plot all previous experiments in blue with 0.5 opacity
        sns.scatterplot(x='std_X', y=WAKE_ACC_COL, data=previous_df, ax=axes[1, 2], 
                        alpha=0.5, color='blue', label='Previous Experiments')
        
        # Plot the latest experiment in orange
        sns.scatterplot(x='std_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 2], 
                        alpha=1.0, color='orange', marker='X', label=f'Latest Exp {last_experiment_hash}')
        
        # Only run regplot on the latest experiment data
        if not last_exp_df.empty:
            sns.regplot(x='std_X', y=WAKE_ACC_COL, data=last_exp_df, ax=axes[1, 2], 
                    scatter=False, color='red')
        
        axes[1, 2].set_title('Wake Accuracy vs Spectrogram Std')
        axes[1, 2].set_xlabel('Spectrogram Standard Deviation')
        axes[1, 2].set_ylabel('Wake Accuracy')
        axes[1, 2].grid(True)
        axes[1, 2].legend()

    fig.tight_layout(pad=0.1)
    # plt.show()
    fig.savefig(str(experiment_results_csv.resolve()).replace(".csv", ".png"), bbox_inches='tight', dpi=200)

if __name__ == "__main__":
    from main import EXPERIMENT_RESULTS_CSV
    from main_dreamt_walch_comp import WALCH_TO_DREAMT_CSV, DREAMT_TO_WALCH_CSV

    # plot(EXPERIMENT_RESULTS_CSV)
    # plot(WALCH_TO_DREAMT_CSV)
    plot(DREAMT_TO_WALCH_CSV)
