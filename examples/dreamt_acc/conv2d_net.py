import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_curve

# Assume your data structure is as follows:
@dataclass
class Preprocessed:
    x: np.ndarray       # (any shape, not used in training here)
    y: np.ndarray       # shape (N,), labels (with -1 to mask)
    x_spec: np.ndarray  # shape (N, 129), the input "image" for the subject

# --- Model Definition ---
def dynamic_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    else:
        return kernel_size // 2

class ConvSegmenterUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvSegmenterUNet, self).__init__()
        self.kernel = (3, 3)
        self.stride = (1, 2)
        pad = dynamic_padding(self.kernel)
        # Encoder: using stride (1,2) so the first dimension (N) remains the same.
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=self.kernel, stride=self.stride, padding=pad)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel, stride=self.stride, padding=pad)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=self.kernel, stride=self.stride, padding=pad)
        # Decoder: mirror the encoder.
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=self.kernel, stride=self.stride, padding=pad, output_padding=(0,0))
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=self.kernel, stride=self.stride, padding=pad, output_padding=(0,0))
        self.dec_deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=self.kernel, stride=self.stride, padding=pad, output_padding=(0,0))
    
    def forward(self, x):
        # x: (B, N, 129) -> add a channel dimension to get (B, 1, N, 129)
        x = x.unsqueeze(1)
        # Encoder
        x = torch.relu(self.enc_conv1(x))  # shape: (B, 32, N, ~65)
        x = torch.relu(self.enc_conv2(x))  # shape: (B, 64, N, ~33)
        x = torch.relu(self.enc_conv3(x))  # shape: (B, 128, N, ~17)
        # Decoder
        x = torch.relu(self.dec_deconv1(x))  # shape: (B, 64, N, ~33)
        x = torch.relu(self.dec_deconv2(x))  # shape: (B, 32, N, ~65)
        x = self.dec_deconv3(x)              # shape: (B, num_classes, N, 129)
        # Collapse the width dimension by averaging over the 129-dimension.
        x = x.mean(dim=3)   # shape: (B, num_classes, N)
        # Permute to get (B, N, num_classes)
        x = x.permute(0, 2, 1)
        return x

# --- LOOCV Training Loop with Specificity at Sensitivity 0.95 ---
def train_loocv(data_list, num_epochs=20, lr=1e-3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    num_folds = len(data_list)
    fold_results = []
    
    for fold in range(num_folds):
        print(f"\nStarting fold {fold+1}/{num_folds}")
        
        # Use subject `fold` as the test set; the rest are training.
        test_subject = data_list[fold]
        train_subjects = [data_list[i] for i in range(num_folds) if i != fold]
        
        # Prepare training tensors:
        # Stack the x_spec and y arrays along a new batch dimension.
        X_train = np.stack([sub.x_spec for sub in train_subjects], axis=0)  # shape: (num_train, N, 129)
        y_train = np.stack([sub.y for sub in train_subjects], axis=0)       # shape: (num_train, N)
        # For the test subject, add a batch dimension.
        X_test = test_subject.x_spec[np.newaxis, ...]  # shape: (1, N, 129)
        y_test = test_subject.y[np.newaxis, ...]         # shape: (1, N)
        
        # Convert numpy arrays to torch tensors.
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
        
        # Initialize model, optimizer, and loss function.
        model = ConvSegmenterUNet(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Training loop for the current fold.
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            # Forward pass: output shape (B, N, 2)
            outputs = model(X_train_tensor)
            # Flatten outputs and targets: (B*N, 2) and (B*N,)
            loss = criterion(outputs.view(-1, 2), y_train_tensor.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        # --- Evaluation on the Test Subject ---
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)  # shape: (1, N, 2)
            # We want the predicted probability for class 1.
            # Flatten predictions and targets:
            y_test_flat = y_test_tensor.view(-1)  # shape: (N,)
            probs_flat = test_outputs[..., 1].view(-1)  # predicted probabilities for class 1
            
            # Create a mask to ignore -1 labels.
            valid_mask = (y_test_flat != -1)
            if valid_mask.sum() == 0:
                print("No valid test labels available; skipping evaluation for this fold.")
                specificity_at_target = None
            else:
                y_true = y_test_flat[valid_mask].cpu().numpy()  # ground truth labels
                probs = probs_flat[valid_mask].cpu().numpy()      # predicted probabilities for class 1
                
                # Compute ROC curve for the positive class (class 1).
                fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)
                target_sensitivity = 0.95
                indices = np.where(tpr >= target_sensitivity)[0]
                if len(indices) == 0:
                    specificity_at_target = 0.0
                    best_threshold = 1.0
                    print(f"Fold {fold+1}: Sensitivity of {target_sensitivity} not reached; specificity set to 0.")
                else:
                    idx = indices[0]
                    specificity_at_target = 1 - fpr[idx]  # specificity = 1 - false positive rate
                    best_threshold = thresholds[idx]
                    print(f"Fold {fold+1} Test: At threshold {best_threshold:.2f}, Sensitivity = {tpr[idx]:.2f}, Specificity = {specificity_at_target:.2f}")
        
        fold_results.append({
            'fold': fold,
            'test_specificity_at_95_sensitivity': specificity_at_target,
            'threshold': best_threshold if valid_mask.sum() > 0 else None
        })
    
    return fold_results

# ---------------------------
# Example Usage:
# Assuming you have a list called `data_list` where each element is a Preprocessed object.
# For demonstration, we create dummy data for 5 subjects.
if __name__ == '__main__':
    num_subjects = 5
    dummy_data = []
    N = 100  # e.g., each subject has 100 segments
    for _ in range(num_subjects):
        x_spec = np.random.randn(N, 129).astype(np.float32)
        # Random labels from {0, 1} with some -1 values as masks.
        y = np.random.choice([0, 1, -1], size=(N,), p=[0.4, 0.5, 0.1]).astype(np.int64)
        dummy_data.append(Preprocessed(x=np.array([]), y=y, x_spec=x_spec))
    
    results = train_loocv(dummy_data, num_epochs=10)
    print("\nLOOCV Results:")
    for res in results:
        print(res)
