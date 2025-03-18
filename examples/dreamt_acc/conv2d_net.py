from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


from examples.dreamt_acc.constants import MASK_VALUE, PSG_MAX_IDX
from examples.dreamt_acc.preprocess import STFT, Preprocessed

# --- Model Definition ---
def dynamic_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    else:
        return kernel_size // 2

class ConvSegmenterUNet(nn.Module):
    def __init__(self, num_classes=2, negative_slope=0.1):
        super(ConvSegmenterUNet, self).__init__()
        self.kernel = (15, 17)
        self.stride = (1, 2)
        pad = dynamic_padding(self.kernel)
        
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=self.kernel, stride=self.stride, padding=pad)
        self.enc_bn1   = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel, stride=self.stride, padding=pad)
        self.enc_bn2   = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=self.kernel, stride=self.stride, padding=pad)
        self.enc_bn3   = nn.BatchNorm2d(128)
        
        # Decoder layers
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=self.kernel, stride=self.stride, 
                                              padding=pad, output_padding=(0,0))
        self.dec_bn1     = nn.BatchNorm2d(64)
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=self.kernel, stride=self.stride, 
                                              padding=pad, output_padding=(0,0))
        self.dec_bn2     = nn.BatchNorm2d(32)
        self.dec_deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=self.kernel, stride=self.stride, 
                                              padding=pad, output_padding=(0,0))
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    
    def forward(self, x):
        # x: (B, N, 129) -> add a channel dimension to get (B, 1, N, 129)
        x = x.unsqueeze(1)
        
        # Encoder: conv -> BN -> LeakyReLU
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.leaky_relu(x)
        
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.leaky_relu(x)
        
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.leaky_relu(x)
        
        # Decoder: deconv -> BN -> LeakyReLU (except final layer)
        x = self.dec_deconv1(x)
        x = self.dec_bn1(x)
        x = self.leaky_relu(x)
        
        x = self.dec_deconv2(x)
        x = self.dec_bn2(x)
        x = self.leaky_relu(x)
        
        x = self.dec_deconv3(x)  # output logits for each class
        
        # Collapse the width dimension by averaging over the 129-dimension.
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

def wasa(model, X_test_tensor, y_test_tensor, wasa_key, WASA_ACC):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)  # shape: (1, 2, N)
        y_test_flat = y_test_tensor.reshape(-1)  # shape: (N,)
        # Get predicted probability for class 1.
        probs_flat = test_outputs[0, 1].reshape(-1) # shape: (N,)
        
        # Create a mask to ignore -1 labels.
        valid_mask = (y_test_flat != -1)
        if valid_mask.sum() == 0:
            print("No valid test labels available; skipping evaluation for this fold.")
            specificity_at_target = None
            best_threshold = None
        else:
            y_true = y_test_flat[valid_mask].cpu().numpy()  # ground truth labels
            probs = probs_flat[valid_mask].cpu().numpy()      # predicted probabilities for class 1
            
            # Compute ROC curve for the positive class.
            fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)
            indices = np.where(tpr >= WASA_ACC)[0]
            if len(indices) == 0:
                specificity_at_target = 0.0
                best_threshold = 1.0
                print(f"Error! Sensitivity of {WASA_ACC} not reached; specificity set to 0.")
            else:
                idx = indices[0]
                specificity_at_target = 1 - fpr[idx]  # specificity = 1 - false positive rate
                sensitivity_at_target = tpr[idx]
                best_threshold = thresholds[idx]
    
    return {
        wasa_key: specificity_at_target,
        'senstivity': sensitivity_at_target,
        'threshold': best_threshold
    }

# --- LOOCV Training Loop with Specificity at Sensitivity 0.95 ---
@dataclass
class TrainingResult:
    idno: str
    fold: int
    logits_threshold: float
    specificity: float
    sensitivity: float
    best_model_path: str
    wasa_result: dict
    test_X: np.ndarray
    test_y: np.ndarray
    sleep_logits: np.ndarray

def train_loocv(data_list: List[Preprocessed], 
                num_epochs=20, lr=1e-3, batch_size=1,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[TrainingResult]:
    print("Training using", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    num_folds = len(data_list)
    fold_results: List[TrainingResult] = []
    scaler = torch.amp.GradScaler()
    for data_subject in data_list:
        # Compute spectrograms for all subjects.
        # this speeds up per-split X_train, X_test computation.
        data_subject.x_spec.compute_specgram()

        # Convert to binary labels: 0, 1, leaving -1 masks as is.
        data_subject.y = np.where(
            data_subject.y > 0,
            1,
            data_subject.y)
    # X_train = np.zeros((num_folds-1, PSG_MAX_IDX, 129), dtype=np.float32)
    # y_train = np.zeros((num_folds-1, PSG_MAX_IDX), dtype=np.int64)
    # X_test = np.zeros((1, PSG_MAX_IDX, 129), dtype=np.float32)
    # y_test = np.zeros((1, PSG_MAX_IDX), dtype=np.int64)
    # default `log_dir` is "runs" - we'll be more specific here
    

    commit_hash = get_git_commit_hash()
    training_dir = Path(os.getcwd()).resolve() / 'dreamt_training_logs' / commit_hash
    writer = SummaryWriter(training_dir)
    report_freq = 5
    WASA_ACC = 0.95
    wasa_key = f'wasa{WASA_ACC}'

    print(f"Training with {num_folds} subjects")
    print(f"Results will appear in {training_dir}")

    for fold in range(num_folds):
        print(f"\nStarting fold {fold+1}/{num_folds}")
        
        # Use subject `fold` as the test set; the rest are training.
        test_subject = data_list[fold]
        train_subjects = [data_list[i] for i in range(num_folds) if i != fold]

        best_model_path = training_dir / f'{test_subject.idno}_best_model_fold.pth'

        # for i, train_subject in enumerate(train_subjects):
        #     X_train[i, :train_subject.x_spec.shape[0]] = train_subject.x_spec
        #     y_train[i, :train_subject.y.shape[0]] = train_subject.y
        
        # # Prepare training tensors:
        # X_test[0, :test_subject.x_spec.shape[0]] = test_subject.x_spec  # shape: (1, N, 129)
        # y_test[0, :test_subject.y.shape[0]] = test_subject.y         # shape: (1, N)
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
        ce_loss = nn.CrossEntropyLoss(ignore_index=MASK_VALUE)


        
        # Training loop for the current fold.
        best_wasa = 0.0
        best_wasa_result = {}
        best_threshold = 0.0
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            # shuffle data
            indices = torch.randperm(X_train_tensor.size(0))
            X_train_tensor = X_train_tensor[indices]
            y_train_tensor = y_train_tensor[indices]
            for batch_idx in range(0, X_train_tensor.size(0), batch_size):
                print_batch = batch_idx // batch_size
                batches = X_train_tensor.size(0) // batch_size
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {print_batch+1}/{batches}')
                # Get batch
                batch_X = X_train_tensor[batch_idx:batch_idx+batch_size]
                batch_y = y_train_tensor[batch_idx:batch_idx+batch_size]
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    # Check for NaNs in input
                    if torch.isnan(batch_X).any():
                        print("Warning: NaN values detected in input batch")
                        
                    outputs = model(batch_X)
                    
                    # Check for NaNs in output
                    if torch.isnan(outputs).any():
                        print("Warning: NaN values detected in model output")
                        
                    loss = ce_loss(outputs, batch_y)
                    
                    # Check for NaNs in loss
                    if torch.isnan(loss):
                        print("Warning: NaN loss detected")
                        continue  # Skip this batch
                
                # print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{X_train_tensor.size(0)}, Loss: {loss.item()}')
                # Backward and optimize with scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                if batch_idx % report_freq == -1 % report_freq:
                    # ...log the running loss
                    writer.add_scalar('training loss',
                                    running_loss / report_freq,
                                    epoch * len(data_list) + batch_idx)
                    running_loss = 0.0
                epoch_wasa_result = wasa(model, X_test_tensor, y_test_tensor, wasa_key, WASA_ACC)
                epoch_wasa = epoch_wasa_result[wasa_key]
                writer.add_scalar(f'training {wasa_key}',
                                epoch_wasa,
                                epoch * len(data_list) + batch_idx)
                if epoch_wasa > best_wasa:
                    print("NEW BEST WASA", epoch_wasa)
                    best_wasa = epoch_wasa
                    best_threshold = epoch_wasa_result['threshold']
                    best_wasa_result = epoch_wasa_result
                    torch.save(model.state_dict(), best_model_path)
                
                # adjust learning rate
                if epoch % 10 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.9
                
        
        # --- Evaluation on the Test Subject ---
        best_threshold = best_wasa_result['threshold']
        specificity_at_target = best_wasa_result[wasa_key]
        tpr = best_wasa_result['senstivity']

        print(f"Fold {fold+1} Test: At threshold {best_threshold:.2f}, Sensitivity = {tpr:.2f}, Specificity = {specificity_at_target:.2f}")

        test_outputs = model(X_test_tensor)[0].cpu().detach().numpy()
        this_fold_result = TrainingResult(
            idno=test_subject.idno,
            fold=fold,
            logits_threshold=best_threshold,
            specificity=specificity_at_target,
            sensitivity=tpr,
            best_model_path=best_model_path,
            wasa_result=best_wasa_result,
            test_X=test_subject.x_spec.specgram,
            test_y=test_subject.y,
            sleep_logits=test_outputs
        )
    
        fold_results.append(this_fold_result)

        writer.add_scalar(f'test specificity at {WASA_ACC} sensitivity',
                        specificity_at_target,
                        fold)
        
        print_histogram([
            f.specificity for f in fold_results
        ], bins=10)
    
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
