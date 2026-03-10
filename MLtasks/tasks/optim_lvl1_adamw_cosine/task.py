"""
Optimizer Comparison: AdamW + Cosine Annealing vs Vanilla SGD

Mathematical Formulation:
- AdamW update rule (decoupled weight decay):
    m_t = β₁ m_{t-1} + (1 - β₁) g_t
    v_t = β₂ v_{t-1} + (1 - β₂) g_t²
    m̂_t = m_t / (1 - β₁^t),   v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - η_t (m̂_t / (√v̂_t + ε) + λ θ_{t-1})

- Cosine Annealing with Warm Restarts:
    η_t = η_min + 0.5 (η_max - η_min)(1 + cos(π T_cur / T_i))

- SGD baseline:
    θ_t = θ_{t-1} - η g_t

Dataset: Synthetic nonlinear regression with 8 features and interaction terms.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_name': 'optim_lvl1_adamw_cosine',
        'series': 'Optimizer Comparison',
        'level': 1,
        'description': 'Compare AdamW+CosineAnnealing vs vanilla SGD on nonlinear regression',
        'model_type': 'mlp_regression',
        'loss_type': 'mse',
        'optimization': 'adamw_cosine_vs_sgd',
        'input_dim': 8,
        'output_dim': 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=1000, batch_size=64, train_ratio=0.8):
    """
    Synthetic nonlinear regression:
        y = sin(x1) + x2*x3 - x4² + 0.5*x5 + noise
    with 8 raw features (some informative, some noise).
    """
    X = np.random.randn(n_samples, 8).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2] - X[:, 3] ** 2
         + 0.5 * X[:, 4] + np.random.normal(0, 0.3, n_samples)).astype(np.float32)

    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mu) / sigma
    y_mu, y_sigma = y.mean(), y.std() + 1e-8
    y = (y - y_mu) / y_sigma

    n_train = int(n_samples * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class RegressionMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(device=None, input_dim=8, hidden_dims=(128, 64)):
    device = device or get_device()
    return RegressionMLP(input_dim, hidden_dims).to(device)


def _train_one_config(model, train_loader, val_loader, device, optimizer, scheduler, epochs, label):
    criterion = nn.MSELoss()
    loss_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + n / len(train_loader))
            running += loss.item()
            n += 1

        avg = running / n
        loss_hist.append(avg)

        val_m = evaluate(model, val_loader, device)
        val_hist.append(val_m['mse'])

        if (epoch + 1) % 20 == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}  train_mse={avg:.6f}  "
                  f"val_mse={val_m['mse']:.6f}  val_r2={val_m['r2']:.4f}")

    return loss_hist, val_hist


def train(model, train_loader, val_loader, device=None, epochs=80, lr=1e-3):
    """Train using AdamW + CosineAnnealingWarmRestarts (primary config)."""
    device = device or get_device()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    loss_hist, val_hist = _train_one_config(
        model, train_loader, val_loader, device, optimizer, scheduler, epochs, 'AdamW+Cosine'
    )
    return {'loss_history': loss_hist, 'val_loss_history': val_hist}


def evaluate(model, data_loader, device=None):
    device = device or get_device()
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X_b, y_b in data_loader:
            preds = model(X_b.to(device))
            all_preds.append(preds.cpu())
            all_tgts.append(y_b)
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_tgts)
    mse = torch.mean((y_pred - y_true) ** 2).item()
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {'mse': mse, 'r2': r2}


def predict(model, X, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return model(X.to(device)).cpu().numpy()


def save_artifacts(model, metrics, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Optimizer Comparison: AdamW+Cosine vs Vanilla SGD")
    print("=" * 60)

    EPOCHS = 80
    LR = 1e-3
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000, batch_size=64
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # --- AdamW + Cosine Annealing ---
    print("\n--- Training with AdamW + CosineAnnealingWarmRestarts ---")
    set_seed(42)
    model_adamw = build_model(device)
    history_adamw = train(model_adamw, train_loader, val_loader, device, epochs=EPOCHS, lr=LR)
    adamw_val = evaluate(model_adamw, val_loader, device)

    # --- Vanilla SGD baseline ---
    print("\n--- Training with Vanilla SGD ---")
    set_seed(42)
    model_sgd = build_model(device)
    sgd_opt = optim.SGD(model_sgd.parameters(), lr=LR)
    sgd_loss, sgd_val_hist = _train_one_config(
        model_sgd, train_loader, val_loader, device, sgd_opt, None, EPOCHS, 'SGD'
    )
    sgd_val = evaluate(model_sgd, val_loader, device)

    print("\n" + "-" * 60)
    print("Comparison Summary")
    print("-" * 60)
    print(f"AdamW+Cosine  -> val_MSE: {adamw_val['mse']:.6f}  val_R2: {adamw_val['r2']:.4f}")
    print(f"Vanilla SGD   -> val_MSE: {sgd_val['mse']:.6f}  val_R2: {sgd_val['r2']:.4f}")

    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0

    if adamw_val['r2'] <= 0.70:
        print(f"FAIL: AdamW val R2 {adamw_val['r2']:.4f} <= 0.70")
        exit_code = 1
    else:
        print(f"PASS: AdamW val R2 > 0.70: {adamw_val['r2']:.4f}")

    if adamw_val['mse'] >= sgd_val['mse']:
        print(f"WARN: AdamW MSE ({adamw_val['mse']:.6f}) not better than SGD ({sgd_val['mse']:.6f})")
    else:
        improvement = (sgd_val['mse'] - adamw_val['mse']) / (sgd_val['mse'] + 1e-8) * 100
        print(f"PASS: AdamW MSE {improvement:.1f}% lower than SGD")

    final_loss = history_adamw['loss_history'][-1]
    init_loss = history_adamw['loss_history'][0]
    if final_loss >= init_loss:
        print(f"FAIL: AdamW loss did not decrease")
        exit_code = 1
    else:
        print(f"PASS: AdamW loss decreased ({init_loss:.6f} -> {final_loss:.6f})")

    adamw_train = evaluate(model_adamw, train_loader, device)
    gap = abs(adamw_train['mse'] - adamw_val['mse'])
    if gap > 0.3:
        print(f"FAIL: Overfitting detected, MSE gap = {gap:.4f}")
        exit_code = 1
    else:
        print(f"PASS: No severe overfitting, MSE gap = {gap:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model_adamw, {
        'adamw_val': adamw_val,
        'sgd_val': sgd_val,
    }, OUTPUT_DIR)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME CHECKS FAILED")
        print("=" * 60)

    sys.exit(exit_code)
