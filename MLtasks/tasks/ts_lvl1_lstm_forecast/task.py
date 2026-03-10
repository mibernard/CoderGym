"""
Time Series Forecasting with LSTM

Mathematical Formulation:
- LSTM Cell:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)          (forget gate)
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)          (input gate)
    ĉ_t = tanh(W_c · [h_{t-1}, x_t] + b_c)       (candidate cell)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t              (cell state)
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)           (output gate)
    h_t = o_t ⊙ tanh(c_t)                          (hidden state)

- Loss: MSE = (1/N) Σ (ŷ_i - y_i)²

Dataset: Synthetic multivariate time series composed of sinusoidal signals
with trend and noise, split into sliding windows for sequence-to-one prediction.
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
        'task_name': 'ts_lvl1_lstm_forecast',
        'series': 'Time Series Forecasting',
        'level': 1,
        'description': 'LSTM-based time series forecasting on synthetic multi-component signal',
        'model_type': 'lstm',
        'loss_type': 'mse',
        'optimization': 'adam',
        'input_dim': 3,
        'output_dim': 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _generate_time_series(n_steps=2000):
    """Generate a synthetic multivariate time series with 3 features and 1 target."""
    t = np.linspace(0, 8 * np.pi, n_steps)
    trend = 0.02 * t
    seasonal = np.sin(t) + 0.5 * np.sin(3 * t)
    noise = np.random.normal(0, 0.1, n_steps)
    target = trend + seasonal + noise

    feat1 = np.sin(t) + np.random.normal(0, 0.05, n_steps)
    feat2 = np.cos(t) + np.random.normal(0, 0.05, n_steps)
    feat3 = np.sin(2 * t) + np.random.normal(0, 0.05, n_steps)

    features = np.column_stack([feat1, feat2, feat3])
    return features, target


def _create_sliding_windows(features, target, window_size=30):
    """Convert raw series into (X, y) pairs using a sliding window."""
    X, y = [], []
    for i in range(len(target) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_dataloaders(batch_size=64, train_ratio=0.8, window_size=30):
    features, target = _generate_time_series(n_steps=2000)

    feat_mean, feat_std = features.mean(axis=0), features.std(axis=0)
    tgt_mean, tgt_std = target.mean(), target.std()
    features = (features - feat_mean) / (feat_std + 1e-8)
    target = (target - tgt_mean) / (tgt_std + 1e-8)

    X, y = _create_sliding_windows(features, target, window_size)
    n_train = int(len(X) * train_ratio)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def build_model(device=None, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
    device = device or get_device()
    model = LSTMForecaster(input_dim, hidden_dim, num_layers, dropout).to(device)
    return model


def train(model, train_loader, val_loader, device=None, epochs=60, lr=1e-3):
    device = device or get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / n_batches
        loss_history.append(avg_train)

        val_metrics = evaluate(model, val_loader, device)
        val_loss_history.append(val_metrics['mse'])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  train_mse={avg_train:.6f}  "
                  f"val_mse={val_metrics['mse']:.6f}  val_r2={val_metrics['r2']:.4f}")

    return {'loss_history': loss_history, 'val_loss_history': val_loss_history}


def evaluate(model, data_loader, device=None):
    device = device or get_device()
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    mse = torch.mean((y_pred - y_true) ** 2).item()
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()

    return {'mse': mse, 'r2': r2, 'mae': mae}


def predict(model, X, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        return model(X).cpu().numpy()


def save_artifacts(model, metrics, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Time Series Forecasting with LSTM")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=64, train_ratio=0.8, window_size=30
    )
    print(f"Training windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")

    print("\nBuilding model...")
    model = build_model(device, input_dim=3, hidden_dim=64, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nTraining for 60 epochs...")
    history = train(model, train_loader, val_loader, device, epochs=60, lr=1e-3)

    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Train  -> MSE: {train_metrics['mse']:.6f}  R2: {train_metrics['r2']:.4f}  MAE: {train_metrics['mae']:.4f}")
    print(f"Val    -> MSE: {val_metrics['mse']:.6f}  R2: {val_metrics['r2']:.4f}  MAE: {val_metrics['mae']:.4f}")

    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0

    if val_metrics['r2'] <= 0.80:
        print(f"FAIL: Validation R2 {val_metrics['r2']:.4f} <= 0.80")
        exit_code = 1
    else:
        print(f"PASS: Validation R2 > 0.80: {val_metrics['r2']:.4f}")

    if val_metrics['mse'] >= 0.15:
        print(f"FAIL: Validation MSE {val_metrics['mse']:.6f} >= 0.15")
        exit_code = 1
    else:
        print(f"PASS: Validation MSE < 0.15: {val_metrics['mse']:.6f}")

    final_train_loss = history['loss_history'][-1]
    initial_train_loss = history['loss_history'][0]
    if final_train_loss >= initial_train_loss:
        print(f"FAIL: Loss did not decrease ({initial_train_loss:.6f} -> {final_train_loss:.6f})")
        exit_code = 1
    else:
        print(f"PASS: Loss decreased ({initial_train_loss:.6f} -> {final_train_loss:.6f})")

    overfit_gap = abs(train_metrics['mse'] - val_metrics['mse'])
    if overfit_gap > 0.1:
        print(f"FAIL: Overfitting detected, MSE gap = {overfit_gap:.4f}")
        exit_code = 1
    else:
        print(f"PASS: No severe overfitting, MSE gap = {overfit_gap:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, OUTPUT_DIR)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME CHECKS FAILED")
        print("=" * 60)

    sys.exit(exit_code)
