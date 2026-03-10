"""
Multi-Task Learning with Shared Encoder

Mathematical Formulation:
- Shared encoder:  h = f_shared(x; θ_shared)
- Task A head:     ŷ_A = g_A(h; θ_A)     (regression)
- Task B head:     ŷ_B = g_B(h; θ_B)     (binary classification)

- Joint loss with task balancing:
    L = w_A · MSE(ŷ_A, y_A) + w_B · BCE(ŷ_B, y_B)

Multi-task learning exploits shared structure between related tasks so that
the encoder learns richer, more generalizable representations. The two tasks
share a common set of input features but predict different targets derived
from the same underlying data-generating process.

Dataset: Synthetic data where
    y_reg   = x₁ + sin(x₂) + x₃² + noise   (regression target)
    y_class = 1[y_reg > median(y_reg)]        (classification target)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_name': 'mtl_lvl1_shared_encoder',
        'series': 'Multi-Task Learning',
        'level': 1,
        'description': 'Multi-task learning: shared encoder with regression + classification heads',
        'model_type': 'mtl_mlp',
        'loss_type': 'mse + bce',
        'optimization': 'adam',
        'input_dim': 10,
        'output_dim': '1 (reg) + 1 (cls)',
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=1500, batch_size=64, train_ratio=0.8):
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y_reg = (X[:, 0] + np.sin(X[:, 1]) + X[:, 2] ** 2
             + 0.3 * X[:, 3] + np.random.normal(0, 0.3, n_samples)).astype(np.float32)
    median_y = np.median(y_reg)
    y_cls = (y_reg > median_y).astype(np.float32)

    mu, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mu) / std
    y_mu, y_std = y_reg.mean(), y_reg.std() + 1e-8
    y_reg = (y_reg - y_mu) / y_std

    n_train = int(n_samples * train_ratio)
    X_tr, X_val = X[:n_train], X[n_train:]
    yr_tr, yr_val = y_reg[:n_train], y_reg[n_train:]
    yc_tr, yc_val = y_cls[:n_train], y_cls[n_train:]

    train_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(yr_tr),
        torch.from_numpy(yc_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(yr_val),
        torch.from_numpy(yc_val),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_tr, X_val, yr_tr, yr_val, yc_tr, yc_val


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MTLModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim)
        enc_out = hidden_dim // 2
        self.reg_head = nn.Sequential(
            nn.Linear(enc_out, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(enc_out, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.encoder(x)
        y_reg = self.reg_head(h).squeeze(-1)
        y_cls = self.cls_head(h).squeeze(-1)
        return y_reg, y_cls


def build_model(device=None, input_dim=10, hidden_dim=128):
    device = device or get_device()
    return MTLModel(input_dim, hidden_dim).to(device)


def train(model, train_loader, val_loader, device=None, epochs=80, lr=1e-3,
          w_reg=1.0, w_cls=1.0):
    device = device or get_device()
    model.to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for X_b, yr_b, yc_b in train_loader:
            X_b = X_b.to(device)
            yr_b, yc_b = yr_b.to(device), yc_b.to(device)

            optimizer.zero_grad()
            pred_reg, pred_cls = model(X_b)
            loss = w_reg * mse_loss(pred_reg, yr_b) + w_cls * bce_loss(pred_cls, yc_b)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1

        loss_hist.append(running / n)
        val_m = evaluate(model, val_loader, device)
        val_loss_hist.append(val_m['total_loss'])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  train_loss={running/n:.4f}  "
                  f"val_loss={val_m['total_loss']:.4f}  "
                  f"val_reg_R2={val_m['reg_r2']:.4f}  val_cls_acc={val_m['cls_accuracy']:.4f}")

    return {'loss_history': loss_hist, 'val_loss_history': val_loss_hist}


def evaluate(model, data_loader, device=None):
    device = device or get_device()
    model.eval()
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    all_reg_pred, all_reg_tgt = [], []
    all_cls_pred, all_cls_tgt = [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for X_b, yr_b, yc_b in data_loader:
            X_b = X_b.to(device)
            yr_b, yc_b = yr_b.to(device), yc_b.to(device)
            pred_reg, pred_cls = model(X_b)
            loss = mse_fn(pred_reg, yr_b) + bce_fn(pred_cls, yc_b)
            total_loss += loss.item()
            n += 1
            all_reg_pred.append(pred_reg.cpu())
            all_reg_tgt.append(yr_b.cpu())
            all_cls_pred.append(pred_cls.cpu())
            all_cls_tgt.append(yc_b.cpu())

    yr_pred = torch.cat(all_reg_pred)
    yr_true = torch.cat(all_reg_tgt)
    yc_logits = torch.cat(all_cls_pred)
    yc_true = torch.cat(all_cls_tgt)

    reg_mse = torch.mean((yr_pred - yr_true) ** 2).item()
    ss_res = torch.sum((yr_true - yr_pred) ** 2).item()
    ss_tot = torch.sum((yr_true - yr_true.mean()) ** 2).item()
    reg_r2 = 1 - ss_res / (ss_tot + 1e-8)

    cls_preds = (torch.sigmoid(yc_logits) >= 0.5).float()
    cls_acc = (cls_preds == yc_true).float().mean().item()

    return {
        'total_loss': total_loss / max(n, 1),
        'reg_mse': reg_mse,
        'reg_r2': reg_r2,
        'cls_accuracy': cls_acc,
        'mse': reg_mse,
        'r2': reg_r2,
    }


def predict(model, X, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        pred_reg, pred_cls = model(X.to(device))
        return {
            'regression': pred_reg.cpu().numpy(),
            'classification': (torch.sigmoid(pred_cls) >= 0.5).long().cpu().numpy(),
        }


def save_artifacts(model, metrics, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Multi-Task Learning: Shared Encoder")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_tr, X_val, yr_tr, yr_val, yc_tr, yc_val = \
        make_dataloaders(n_samples=1500, batch_size=64)
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}")
    print(f"Class balance (train): {yc_tr.mean():.2f} positive")

    print("\nBuilding model...")
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nTraining for 80 epochs...")
    history = train(model, train_loader, val_loader, device, epochs=80, lr=1e-3)

    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)
    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print(f"Train -> reg_MSE: {train_m['reg_mse']:.4f}  reg_R2: {train_m['reg_r2']:.4f}  "
          f"cls_acc: {train_m['cls_accuracy']:.4f}")
    print(f"Val   -> reg_MSE: {val_m['reg_mse']:.4f}  reg_R2: {val_m['reg_r2']:.4f}  "
          f"cls_acc: {val_m['cls_accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0

    if val_m['reg_r2'] <= 0.70:
        print(f"FAIL: Regression R2 {val_m['reg_r2']:.4f} <= 0.70")
        exit_code = 1
    else:
        print(f"PASS: Regression R2 > 0.70: {val_m['reg_r2']:.4f}")

    if val_m['cls_accuracy'] <= 0.80:
        print(f"FAIL: Classification accuracy {val_m['cls_accuracy']:.4f} <= 0.80")
        exit_code = 1
    else:
        print(f"PASS: Classification accuracy > 0.80: {val_m['cls_accuracy']:.4f}")

    final_loss = history['loss_history'][-1]
    init_loss = history['loss_history'][0]
    if final_loss >= init_loss:
        print(f"FAIL: Joint loss did not decrease")
        exit_code = 1
    else:
        print(f"PASS: Joint loss decreased ({init_loss:.4f} -> {final_loss:.4f})")

    reg_gap = abs(train_m['reg_mse'] - val_m['reg_mse'])
    if reg_gap > 0.3:
        print(f"FAIL: Regression overfitting, MSE gap = {reg_gap:.4f}")
        exit_code = 1
    else:
        print(f"PASS: No severe regression overfitting, MSE gap = {reg_gap:.4f}")

    cls_gap = abs(train_m['cls_accuracy'] - val_m['cls_accuracy'])
    if cls_gap > 0.10:
        print(f"FAIL: Classification overfitting, acc gap = {cls_gap:.4f}")
        exit_code = 1
    else:
        print(f"PASS: No severe classification overfitting, acc gap = {cls_gap:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, val_m, OUTPUT_DIR)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME CHECKS FAILED")
        print("=" * 60)

    sys.exit(exit_code)
