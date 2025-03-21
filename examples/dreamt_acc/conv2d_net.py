from dataclasses import dataclass
from hashlib import sha256
from logging import warning
import os
from pathlib import Path
import sys
from typing import List
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve
import seaborn as sns


from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


from examples.dreamt_acc.constants import MASK_VALUE, PSG_MAX_IDX
from examples.dreamt_acc.preprocess import STFT, Preprocessed


import torch
import torch.nn as nn
from typing import List

# --- Model Definition ---
def dynamic_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    else:
        return kernel_size // 2

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.1):
        super(UNetEncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, negative_slope=0.1, apply_bn=True):
        super(UNetDecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.apply_bn = apply_bn
        if apply_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.deconv(x)
        if self.apply_bn:
            x = self.bn(x)
            x = self.leaky_relu(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, channels: List[int], kernel_size, stride, padding, negative_slope=0.1):
        super(UNetEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for out_channels in channels:
            self.blocks.append(UNetEncoderBlock(current_channels, out_channels, kernel_size, stride, padding, negative_slope))
            current_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, channels: List[int], kernel_size, stride, padding, output_padding, negative_slope=0.1, final_bn=True):
        super(UNetDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            # Don't apply batch norm on the final layer if specified
            apply_bn = True if (i < len(channels) - 1 or final_bn) else False
            self.blocks.append(UNetDecoderBlock(current_channels, out_channels, kernel_size, stride, padding, output_padding, negative_slope, apply_bn))
            current_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvSegmenterUNet(nn.Module):
    def __init__(self, num_classes=2, negative_slope=0.1):
        super(ConvSegmenterUNet, self).__init__()
        self.kernel = (15, 5)
        self.stride = (1, 2)
        self.output_padding = (0, 0)
        pad = dynamic_padding(self.kernel)
        
        # Initialize channel dimensions for layers
        self.channels = [8, 16, 32]
        
        self.first_bn = nn.BatchNorm2d(1)
        
        # Encoder 1
        self.encoder1 = UNetEncoder(1, self.channels, self.kernel, self.stride, pad, negative_slope)
        
        # Decoder 1
        dec1_channels = list(reversed(self.channels[:-1])) + [self.channels[-1]]  # Output matches E2 input
        self.decoder1 = UNetDecoder(self.channels[-1], dec1_channels, self.kernel, self.stride, pad, self.output_padding, negative_slope)
        
        # Encoder 2
        self.encoder2 = UNetEncoder(self.channels[-1], self.channels, self.kernel, self.stride, pad, negative_slope)
        
        # Decoder 2
        dec2_channels = list(reversed(self.channels[:-1])) + [self.channels[0]]
        self.decoder2 = UNetDecoder(self.channels[-1], dec2_channels, self.kernel, self.stride, pad, self.output_padding, negative_slope)
        
        # Final classifier
        self.final_conv = nn.Conv2d(self.channels[0], num_classes, kernel_size=(1, 1))
    
    def forward(self, x):
        # x: (B, N, 129) -> add a channel dimension to get (B, 1, N, 129)
        x = x.unsqueeze(1)
        x = self.first_bn(x)
        
        # Encoder 1 (E1)
        x_e1 = self.encoder1(x)
        
        # Decoder 1 (D1)
        x = self.decoder1(x_e1)
        
        # Encoder 2 (E2)
        x = self.encoder2(x)
        
        # Add skip connection from E1 to D2 input
        x = x + x_e1
        
        # Decoder 2 (D2)
        x = self.decoder2(x)
        
        # Final classifier
        x = self.final_conv(x)
        
        # Collapse the width dimension by averaging over the 129-dimension
        x = x.mean(dim=3)   # shape: (B, num_classes, N)
        
        return x


import subprocess

def get_git_commit_hash():
    """Return the full commit SHA of the current Git HEAD."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        return commit_hash.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        # return a hash of the error message
        return sha256(str(e).encode()).hexdigest()

def write_folds(fold_results, file_path):
    """Write fold results to a CSV file."""
    import csv

    write_header = not file_path.exists()
    
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = TrainingResult.get_header()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        for result in fold_results:
            writer.writerow(result.to_row())

def print_histogram(data, bins: int=10):
    """prints a text histogram into the terminal"""
    data = np.array(data)
    bins_arr = np.linspace(0, 1, bins+1)
    hist, bin_edges = np.histogram(data, bins=bins_arr)
    pos_prepend = ""
    if bin_edges[0] < 0:
        pos_prepend = " "
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}: {'#' * hist[i]}")
    print(f"min: {data.min():0.3f} max: {data.max():0.3f} median: {np.median(data):0.3f} std: {data.std():0.3f}")

# --- WASA Calculation ---

@dataclass
class WASAResult:
    wake_acc: float
    sleep_acc: float
    threshold: float

def true_pos_neg_rates_from_threshold(y_true, y_pred, threshold):
    """Calculate the true positive rate and true negative rate given a threshold.
    
    For us, true positives tend to be true sleeps, and true negatives are true wakes."""
    y_pred_binary = (y_pred > threshold).astype(int)
    # tn, fp, fn, tp = np.bincount(y_true * 2 + y_pred_binary, minlength=4)
    # write it this way, which is compatible with -1 labels
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    # if no examples, perfect accuracy
    tpr = 1.0
    tnr = 1.0
    if (tp + fn):
        tpr = tp / (tp + fn)
    if (fp + tn):
        tnr = tn / (fp + tn)
    return tpr,tnr 


def wasa(model, X_test_tensor, y_test_tensor, target_sleep_acc) -> WASAResult:
    """Wake Accuracy when Sleep Accuracy is approx WASA_ACC.
    Returns the specificity at the target sensitivity, and the best threshold.

    NB: the meaning of specificity is inverted when wake is class 1, or if we mix outputs of sleep logits with wake as class 1.
    To remove confusion, we use the term WASA instead of the more common Sensitivity at Specificity.
    """
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)  # shape: (B, 2, N)
        y_test_flat = y_test_tensor  # shape: (B, N,)
        # Get predicted probability for class 1.
        
        # Create a mask to ignore -1 labels.
        # valid_mask = (y_test_flat != -1)
        # if valid_mask.sum() == 0:
        #     warning("No valid test labels available; skipping evaluation for this fold.")
        #     return WASAResult(wake_acc=0.0, sleep_acc=0.0, threshold=0.0)

        y_true = y_test_flat.cpu().numpy()  # ground truth labels
        
        # raw_outputs = torch.softmax(test_outputs, dim=1)
        raw_outputs = test_outputs
        valid_outputs = raw_outputs[:, 1].cpu().numpy()  # shape: (B, N,)

        # Check for special cases
        # Case 1: All sleep samples
        default = WASAResult(wake_acc=1.0, sleep_acc=1.0, threshold=0.5)
        if np.all(y_true == 1):
            return default
        
        # Case 2: All wake samples  
        if np.all(y_true == 0):
            return default

        # Use a binary search on threshold, starting halfway between probs.max() and probs.min()
        # Note that we're using logits potentially so we don't assume probs are in [0, 1].
        # print("=== WASA Calculation === ... valid shape: ", valid_outputs.shape)
        valid_min = valid_outputs.min()
        valid_max = valid_outputs.max()
        # print(valid_outputs.min(), "---", valid_outputs.max())
        # print(valid_outputs.mean(), "+/-", valid_outputs.std())
        # lower = raw_outputs.min()
        # upper = raw_outputs.max()
        lower = valid_min
        upper = valid_max
        best_sleep_acc = 0.0
        threshold = 0
        tol = min((upper - lower) / 100, 0.0001)
        binary_search_iterations = 0
        max_iterations = 50
        while (abs(best_sleep_acc - target_sleep_acc) > tol) \
            and (binary_search_iterations < max_iterations) \
                and (lower < upper): # once ==, stop
            binary_search_iterations += 1
            new_threshold = (lower + upper) / 2
            if abs(new_threshold - threshold) < tol:
                break
            threshold = new_threshold
            # sleep, wake because true positive is true sleep
            sleep_acc, wake_acc = true_pos_neg_rates_from_threshold(y_true, valid_outputs, threshold)

            if sleep_acc >= target_sleep_acc + tol:
                # sleep accuracy too high, want to classify more sleep as wake
                # since we score SLEEP (1) when sleep_proba > threshold, we want to increase threshold
                lower = threshold  # threshold = (lower + upper) / 2 will be higher next time
            if sleep_acc <= target_sleep_acc - tol:
                # scoring too much sleep as wake, decrease threshold to get more sleep
                upper = threshold
            
            if (abs(best_sleep_acc - target_sleep_acc) > abs(sleep_acc - target_sleep_acc)):
                best_sleep_acc = sleep_acc 
        
        best_threshold = (lower + upper) / 2
        sleep_acc, wake_acc = true_pos_neg_rates_from_threshold(y_true, valid_outputs, best_threshold)
        # print("Declaring victory with sleep accuracy", sleep_acc, "at threshold", best_threshold)
        # print(f"This gives a wake acc of {wake_acc}. {binary_search_iterations} iters taken.")
            
        
    return WASAResult(wake_acc=wake_acc, sleep_acc=sleep_acc, threshold=best_threshold)
    

# --- LOOCV Training Loop with Specificity at Sensitivity 0.95 ---
@dataclass
class TrainingResult:
    idno: str
    experiment_hash: str
    fold: int
    logits_threshold: float
    wake_acc: float
    sleep_acc: float
    wasa_result: WASAResult
    test_X: np.ndarray
    test_y: np.ndarray
    sleep_logits: np.ndarray
    max_X: float = 0.0
    min_X: float = 0.0
    mean_X: float = 0.0
    median_X: float = 0.0
    std_X: float = 0.0

    @classmethod
    def experiment_id_column(cls) -> str:
        return 'experiment_hash'

    @classmethod
    def id_column(cls) -> str:
        return 'idno'
    
    @classmethod
    def wake_acc_column(cls) -> str:
        return 'wake_acc'

    def to_row(self):
        return {
            self.id_column(): self.idno,
            self.experiment_id_column(): self.experiment_hash,
            'fold': self.fold,
            'logits_threshold': self.logits_threshold,
            self.wake_acc_column(): self.wake_acc,
            'sleep_acc': self.sleep_acc,
            'max_X': self.max_X,
            'min_X': self.min_X,
            'mean_X': self.mean_X,
            'median_X': self.median_X,
            'std_X': self.std_X,
        }
    
    @classmethod
    def from_csv(cls, row):
        return cls(
            idno=row.get('idno'),
            experiment_hash=row.get('experiment_hash'),
            fold=row.get('fold'),
            logits_threshold=row.get('logits_threshold'),
            wake_acc=row.get('wake_acc'),
            sleep_acc=row.get('sleep_acc'),
            max_X=row.get('max_X'),
            min_X=row.get('min_X'),
            mean_X=row.get('mean_X'),
            median_X=row.get('median_X'),
            std_X=row.get('std_X')
        )

    @classmethod
    def get_header(cls):
        return  [
            'idno',
            'experiment_hash',
            'fold',
            'logits_threshold',
            'wake_acc', 
            'sleep_acc',
            'max_X',
            'min_X',
            'mean_X',
            'median_X',
            'std_X'
        ]


def softmax(x, axis=1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_value_for_vector(logits_val: float, vector: np.ndarray):
    """When you compute a threshold with respect to logits, you need to convert it to a probability value.
    This function does that, by inverting the softmax.
    """
    return np.exp(logits_val - np.sum(np.exp(vector)))

def make_beautiful_specgram_plot(
        prepro_x_y: Preprocessed,
        training_res: TrainingResult = None,
        staging: bool = False,
        from_logits: bool = True):
    N_ROWS = 1
    if staging:
        N_ROWS += 1
    if training_res is not None:
        N_ROWS += 1
    fig, ax = plt.subplots(nrows=N_ROWS, figsize=(20, 10), sharex=True)
    fig.tight_layout(w_pad=2.0)
    if prepro_x_y.x_spec is None:
        prepro_x_y.compute_specgram()
    prepro_x_y.x_spec.plot(ax[0])
    print("Spec shape:", prepro_x_y.x_spec.shape)

    y_plot = prepro_x_y.y
    if staging:
        sns.lineplot(x=np.arange(len(y_plot)), y=y_plot, ax=ax[1])
        ax[1].set_xlim(0, len(y_plot))
        ax[1].set_yticks([-1, 0, 1, 2, 3])
        ax[1].set_yticklabels(['Missing', 'W', 'Light', 'Deep', 'REM'])
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Sleep Stage')

    if training_res is not None:
        wake_sleep_proba = softmax(training_res.sleep_logits) if from_logits else training_res.sleep_logits
        sleep_proba = wake_sleep_proba[1]


        sleep_plot_x = np.arange(len(sleep_proba))
        sns.lineplot(x=sleep_plot_x, y=training_res.test_y, ax=ax[-1])

        # Plot the sleep probabilities
        # The scale on these changes a lot; split off a separate axis that shares the x.
        # useful for debugging to have htis respond to the data rather than our belief about the model outputs
        ax_res_rev = ax[-1].twinx()
        sns.lineplot(
            x=sleep_plot_x,
            y=sleep_proba,
            ax=ax_res_rev,
            color='tab:orange',
            # label='(Unscaled) Sleep Logits/Probas')
            label=f'WASA{100 * training_res.sleep_acc:.0f}={training_res.wake_acc:.3f}')
        # threshold_proba = training_res.logits_threshold
        # ax_res_rev.axhline(threshold_proba, 
        #                    linestyle=":", 
        #                    color='tab:orange', 
        #                    linewidth=0.5, 
        #                    label=f'WASA{100 * training_res.sleep_acc:.0f}={training_res.wake_acc:.3f}, t = {threshold_proba:.3f}')

        ax[-1].set_xlim(sleep_plot_x[0], sleep_plot_x[-1])
        missing_y_value = -0.1
        ax[-1].set_ylim(missing_y_value, 1.1)
        ax[-1].set_yticks([missing_y_value, 0, 1])
        ax[-1].set_yticklabels(['Missing', 'Wake', 'Sleep'])
        ax_res_rev.legend()
    return fig, ax


def train_loocv(data_list: List[Preprocessed], 
                experiment_results_csv: Path,
                num_epochs=20, lr=1e-3, batch_size=1,
                min_spec_max: float = 0.1,  # we will filter out subjects with specgram max < this. Excludes poor quality data.
                plot=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[TrainingResult]:
    print("Training using", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    fold_results: List[TrainingResult] = []
    scaler = torch.amp.GradScaler()
    maxes = []
    for data_subject in data_list:
        # Compute spectrograms for all subjects.
        # this speeds up per-split X_train, X_test computation.
        data_subject.x_spec.compute_specgram(normalization_window_idx=None,
                                             freq_max=10)
        print("Frequency filtered specgram size: ", data_subject.x_spec.specgram.shape)
        spec_max = data_subject.x_spec.specgram.max()
        if spec_max < min_spec_max:
            print(f"Skipping subject {data_subject.idno} with specgram max {spec_max:.3f}")
            

        maxes.append(data_subject.x_spec.specgram.max())

        # Convert to binary labels: 0, 1, leaving -1 masks as is.
        data_subject.y = np.where(
            data_subject.y > 0,
            1,
            data_subject.y)
    
    maxes_keep_idx = [i for i, m in enumerate(maxes) if m >= min_spec_max]
    data_list = [data_list[i] for i in maxes_keep_idx]
    maxes = [maxes[i] for i in maxes_keep_idx]
    num_folds = len(data_list)
    fig, ax = plt.subplots(ncols=1, figsize=(10, 5))
    sns.histplot(maxes, bins=10, ax=ax)
    ax.set_title("Maxes")
    fig.savefig("max_min_hist.png")

    commit_hash = get_git_commit_hash()
    training_dir = experiment_results_csv.parent / 'dreamt_training_logs' / commit_hash
    writer = SummaryWriter(training_dir)
    report_freq = 5
    WASA_ACC = 0.95
    wasa_key = f'wasa{WASA_ACC}'

    print(f"Training with {num_folds} subjects")
    print(f"Results will appear in {training_dir}")

    # print a description of the model
    print(ConvSegmenterUNet())

    fold_tqdm = tqdm(range(num_folds))
    for fold in fold_tqdm:
        fold_str = f"{commit_hash} Fold {fold+1}/{num_folds}"
        fold_tqdm.set_description_str(f"\nSTART OF\n{fold_str}")
        
        fold_test_spec_max = maxes[fold]
        
        # Use subject `fold` as the test set; the rest are training.
        test_subject = data_list[fold]
        train_subjects = [data_list[i] for i in range(num_folds) if i != fold]

        best_model_path = training_dir / f'{test_subject.idno}_best_model_fold.pth'

        X_train = np.array([
            train_subject.x_spec.specgram
            for train_subject in train_subjects
        ], dtype=np.float32)
        y_train = np.array([
            train_subject.y
            for train_subject in train_subjects
        ], dtype=np.int64)
        X_test = np.array([
            test_subject.x_spec.specgram
        ], dtype=np.float32)
        y_test = np.array([test_subject.y], dtype=np.int64)
        
        # Convert numpy arrays to torch tensors.
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
        
        # Initialize model, optimizer, and loss function.
        model = ConvSegmenterUNet(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        flat_y = y_train.flatten()
        valid_y = flat_y[flat_y != MASK_VALUE]
        balancing_weights = torch.tensor(
            [1.0, 1.0],
            dtype=torch.float32,
            device=device)

        ce_loss = nn.CrossEntropyLoss(
            ignore_index=MASK_VALUE,
            weight=balancing_weights)


        
        # Training loop for the current fold.
        best_wasa = 0.0
        best_wasa_result: WASAResult
        best_threshold = 0.0
        for epoch in range(num_epochs):
            running_loss = 0.0
            # shuffle data
            indices = torch.randperm(X_train_tensor.size(0))
            epoch_X = X_train_tensor[indices]
            epoch_y = y_train_tensor[indices]
            # epoch_X = X_train_tensor
            # epoch_y = y_train_tensor
            batch_tqdm = tqdm(range(0, epoch_X.size(0), batch_size))
            for batch_idx in batch_tqdm:
                print_batch = batch_idx // batch_size + 1
                print_n_batches = epoch_X.size(0) // batch_size + 1
                batch_tqdm.set_description_str(f'{fold_str} Batch {print_batch}/{print_n_batches}')
                # Get batch
                batch_X = epoch_X[batch_idx:batch_idx+batch_size]
                batch_y = epoch_y[batch_idx:batch_idx+batch_size]
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    # Check for NaNs in input
                    if torch.isnan(batch_X).any():
                        print("Warning: NaN values detected in input batch")
                        
                    outputs = model(batch_X)
                    # sig_outputs = torch.softmax(outputs, dim=1)
                    sig_outputs = outputs
                    
                    # Check for NaNs in output
                    if torch.isnan(sig_outputs).any():
                        print("Warning: NaN values detected in model output")
                        
                    loss = ce_loss(sig_outputs, batch_y)

                    print(" Loss", loss.item())
                    
                    # Check for NaNs in loss
                    if torch.isnan(loss):
                        print("Warning: NaN loss detected")
                        continue  # Skip this batch
                
                # Backward and optimize with scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                if (batch_idx + 1) % report_freq == 0:
                    # ...log the running loss
                    writer.add_scalar('training loss',
                                    running_loss / report_freq,
                                    epoch * len(data_list) + batch_idx)
                    running_loss = 0.0
                epoch_wasa_result = wasa(model, X_test_tensor, y_test_tensor, WASA_ACC)
                epoch_wasa = epoch_wasa_result.wake_acc
                writer.add_scalar(f'training {wasa_key}',
                                epoch_wasa,
                                epoch * len(data_list) + batch_idx)
                if epoch_wasa > best_wasa:
                    print("NEW BEST WASA", epoch_wasa)
                    best_wasa = epoch_wasa
                    best_threshold = epoch_wasa_result.threshold
                    best_wasa_result = epoch_wasa_result
                    torch.save(model.state_dict(), best_model_path)
        
        # --- Evaluation on the Test Subject ---
        best_threshold = best_wasa_result.threshold
        wake_acc = best_wasa_result.wake_acc
        sleep_acc = best_wasa_result.sleep_acc

        print(f"Fold {fold+1} Test: At threshold {best_threshold:.2f}, Sleep Acc = {sleep_acc:.2f}, Wake Acc = {wake_acc:.2f}, Spec max: {fold_test_spec_max:.3f}")
        

        test_outputs = model(X_test_tensor)[0].cpu().detach().numpy()
        this_fold_result = TrainingResult(
            idno=test_subject.idno,
            experiment_hash=commit_hash,
            fold=fold,
            logits_threshold=best_threshold,
            wake_acc=wake_acc,
            sleep_acc=sleep_acc,
            wasa_result=best_wasa_result,
            test_X=test_subject.x_spec.specgram,
            test_y=test_subject.y,
            sleep_logits=test_outputs
        )
    
        fold_results.append(this_fold_result)
        write_folds(fold_results, experiment_results_csv)

        plt.close()
        fig, max_vs_wasa_ax = plt.subplots(figsize=(10, 5))
        fold_maxes = maxes[:fold+1]
        fold_wasas = [f.wake_acc for f in fold_results]
        sns.scatterplot(x=fold_maxes, y=fold_wasas, ax=max_vs_wasa_ax)
        sns.regplot(x=fold_maxes, y=fold_wasas, ax=max_vs_wasa_ax, order=2)
        max_vs_wasa_ax.set_xlabel('Spectrogram Maximum Value')
        max_vs_wasa_ax.set_ylabel('Wake Accuracy')
        fig.savefig(training_dir / 'max_vs_wasa.png')

        writer.add_scalar(f'test specificity at {WASA_ACC} sensitivity',
                        wake_acc,
                        fold)
        
        print_histogram([
            f.wake_acc for f in fold_results
        ], bins=10)
        if plot:
            fig, ax = make_beautiful_specgram_plot(test_subject, this_fold_result, from_logits=False)
            plot_dir = training_dir / f'{test_subject.idno}_result.png'
            plt.savefig(plot_dir, dpi=300)
            plt.close(fig)
        
        del model, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor
    
    return fold_results

# ---------------------------
# Example Usage:
# TODO: rewrite to fit new shapes
# if __name__ == '__main__':
#     num_subjects = 5
#     dummy_data = []
#     N = 100  # e.g., each subject has 100 segments
#     for _ in range(num_subjects):
#         x_spec = STFT(f=None, t=None, Zxx=np.random.randn(N, 129).astype(np.float32))
#         # Random labels from {0, 1} with some -1 values as masks.
#         y = np.random.choice([0, 1, -1], size=(N,), p=[0.4, 0.5, 0.1]).astype(np.int64)
#         dummy_data.append(Preprocessed(x=np.array([]), y=y, x_spec=x_spec))
    
#     results = train_loocv(dummy_data, num_epochs=10)
#     print("\nLOOCV Results:")
#     for res in results:
#         print(res)
