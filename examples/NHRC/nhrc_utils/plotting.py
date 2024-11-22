import os
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt, rcParams
import numpy as np
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import auc, roc_curve

from examples.NHRC.nhrc_utils.analysis import ACCURACY_COLUMN, ID_COLUMN, THRESHOLD, WASA_COLUMN, compute_sample_weights

COLOR_PALETTE = sns.color_palette("colorblind")
rcParams['font.family'] = 'Helvetica'
rcParams['font.size'] = 12  # Set a global font size


def tri_plot_metrics(
        evaluations_df: List[tuple], 
        accuracy_column: str,
        wasa_column: str,
        auroc_column: str,
        save_dir: Path | None = None, 
        PLOT_TITLE: str = "Metrics", 
        axs: List[plt.Axes] | None = None, 
        axs_set_name: str | None = None, 
        filename: str | None = None,
    ):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig = axs[0].figure

    metrics = {
        'sw_accuracy': evaluations_df[accuracy_column],
        'auc': evaluations_df[auroc_column],
        'wasa': evaluations_df[wasa_column]
    }

    metrics_dimensions = {
        'sw_accuracy': 'abs(pred - true) sleep, minutes',
        'auc': 'AUC',
        'wasa': wasa_column}
    metrics_xlabels = {
        'sw_accuracy': 'minutes predicted - true sleep',
        'auc': 'AUC',
        'wasa': wasa_column}
    metric_colors = {'sw_accuracy': COLOR_PALETTE[4], 'auc': COLOR_PALETTE[1], 'wasa': COLOR_PALETTE[2]}

    if axs_set_name is not None:
        axs[0].set_title(axs_set_name)

    for metric, ax in zip(metrics_dimensions.keys(), axs):
        sns.histplot(
            metrics[metric],
            bins=20,
            stat='percent',
            kde=True,
            color=metric_colors[metric],
            ax=ax)
        metric_mean = np.mean(abs(metrics[metric]))
        ax.axvline(metric_mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean {metrics_dimensions[metric]}: {metric_mean:.3f}")
        # ax.set_title(metrics_dimensions[metric])
        ax.set_xlabel(metrics_xlabels[metric])
        if metric != 'sw_accuracy':
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 20)
        else:
            ax.set_xlim(-30, 30)
            ax.set_ylim(0, 20)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Customize the remaining spines (left and bottom)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.set_ylabel('% Density')
        ax.legend()

    fig.suptitle(PLOT_TITLE,
                 fontsize=20)
    fig.tight_layout()
    if save_dir is not None:
        if filename is None:
            filename = 'tri_plot_metrics.png'
        fig.savefig(save_dir.joinpath(filename),
                    # dpi=300,
                    bbox_inches='tight')

def add_spectrogram(ax, specgram):
    ax.imshow(specgram.T, aspect='auto', origin='lower', vmax=20, vmin=-20)
    # Create a secondary y-axis on the right and set its label
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(f'Specgram')
    ax.set_ylabel('Frequency')

def add_smartmap_inputs(ax, input_data):
    input_t = input_data.T
    INPUT_EPOCHS = input_data.shape[0]
    ax.stackplot(range(INPUT_EPOCHS), softmax(input_t, axis=0))
    ax.set_xlim(0, INPUT_EPOCHS)
    # Create a secondary y-axis on the right and set its label
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(f'SmartMap Training Input')
    ax.set_ylabel('Value')
    ax.set_xlabel('Time (epochs)')

def add_lr_inputs(ax, input_data):
    cnn_input_width = input_data.shape[0]
    ax.plot(range(cnn_input_width), input_data)
    ax.set_xlim(0, cnn_input_width)
    # Create a secondary y-axis on the right and set its label
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(f'Actigraphy')
    ax.set_ylabel('Value')
    ax.set_xlabel('Time (epochs)')

def add_hypnogram(ax, binary_pred_proba, true_labels, naive_prediction, model_type, threshold: float | None = None):
    binary_label = 'SmartMap Prediction' if model_type == 'smartmap' else 'LR Prediction'
    ax.plot(binary_pred_proba, label=binary_label)
    if naive_prediction is not None:
        ax.plot(naive_prediction, label='Naive: 1 - P(wake)')

    sample_weight = compute_sample_weights(true_labels)
    masked_true_labels = np.where(sample_weight, true_labels, -1)

    ax.plot(masked_true_labels, label='Actual', linestyle='--')
    if threshold is not None:
        ax.axhline(threshold, color='gold', linestyle='--', label='Threshold: {:.2f}'.format(threshold))
    ax.set_xlim(0, len(binary_pred_proba))
    ax.set_ylabel('Probability of Sleep')
    # Create a secondary y-axis on the right and set its label
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(f'Hypnogram')
    ax.legend()

    return masked_true_labels, sample_weight


def add_roc(ax, binary_pred_proba, true_labels, sample_weight, naive_prediction: np.ndarray | None = None):
    color_palette = sns.color_palette("colorblind")
    masked_true_labels = np.where(sample_weight, true_labels, 0)
    fpr, tpr, thresholds = roc_curve(masked_true_labels, binary_pred_proba, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    roc_auc_naive = None
    if naive_prediction is not None:
        # Plot the ROC curve from naive_prediction
        fpr_naive, tpr_naive, _ = roc_curve(masked_true_labels, naive_prediction, sample_weight=sample_weight)
        roc_auc_naive = auc(fpr_naive, tpr_naive)
        ax.plot(fpr_naive, tpr_naive, color=color_palette[2], lw=2, label=f'Naive ROC curve (area = {roc_auc_naive:.2f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Wake Predicted as Sleep')
    ax.set_ylabel('Sleep Predicted as Sleep')
    ax.legend(loc="lower right")

    return roc_auc, roc_auc_naive

def ranked_debug_plots(df, eval_data, true_labels, predictors, evaluator, specgrams, filename_suffix: str, model_type: str, split_names: List[str] = None, saveto: Path = None, sortby: str = ACCURACY_COLUMN, sort_ascending: bool = False):
    indices = df.sort_values(sortby, ascending=sort_ascending).index
    aurocs = []
    

    for rank, idx in enumerate(indices):
        if idx >= len(eval_data):
            continue
        row = df.loc[idx]
        split_name = row[ID_COLUMN]
        threshold = row[THRESHOLD]
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        fig.subplots_adjust(hspace=0, left=0)
        
        # Plot the specgram
        add_spectrogram(axs[0], specgrams[idx])
        
        # Plot the CNN training input as a stacked area plot
        if model_type == 'smartmap':
            add_smartmap_inputs(axs[1], eval_data[idx].squeeze(), )
        elif model_type == 'lr':
            add_lr_inputs( axs[1], eval_data[idx].squeeze(),)
        
        # Plot the final output along with the correct value

        binary_pred_proba = evaluator(predictors[idx], eval_data[idx]).squeeze()
        naive_prediction = 1 - eval_data[idx][:, 0] if model_type == 'smartmap' else None
        masked_true_labels, sample_weight = add_hypnogram(
            axs[2],
            binary_pred_proba,
            true_labels[idx],
            naive_prediction,
            model_type,
            threshold,
        )

        auroc, *_ = add_roc(
            ax=axs[3],
            binary_pred_proba=binary_pred_proba, 
            true_labels=true_labels[idx], 
            sample_weight=sample_weight, 
            naive_prediction=naive_prediction
        )

        aurocs.append(auroc)

        fig.suptitle(f'#{rank} {split_name} {ACCURACY_COLUMN}: {df.loc[idx, ACCURACY_COLUMN]} {WASA_COLUMN}: {df.loc[idx, WASA_COLUMN]}',
                    fontsize=16, fontweight='bold')
        
        fig.tight_layout(pad=0.1)
        if saveto is None:
            saveto = Path(os.getcwd())
        plt.savefig(saveto.joinpath(f"{split_name}_rank_{rank}_{filename_suffix}.png"))
        plt.close()

    sns.histplot(aurocs, bins=20, kde=True, color='blue', ax=plt.gca(), )
    plt.xlabel('AUROC')
    plt.ylabel('Count')
    plt.title('Distribution of AUROCs')
    plt.show()