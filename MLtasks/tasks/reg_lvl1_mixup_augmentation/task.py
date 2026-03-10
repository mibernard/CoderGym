"""
Mixup Data Augmentation for Classification

Mathematical Formulation:
- Mixup creates virtual training examples by linear interpolation:
    x̃ = λ x_i + (1 − λ) x_j
    ỹ = λ y_i + (1 − λ) y_j
  where λ ~ Beta(α, α), and (x_i, y_i), (x_j, y_j) are drawn at random
  from the training set.

- The mixed loss is:
    L(θ) = λ CE(f_θ(x̃), y_i) + (1 − λ) CE(f_θ(x̃), y_j)

- This acts as a form of data-dependent regularization, encouraging the model
  to produce linear predictions between training examples and improving
  generalization, especially on small or noisy datasets.

Dataset: Synthetic 6-class classification with overlapping decision boundaries
to demonstrate regularization benefit.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_name': 'reg_lvl1_mixup_augmentation',
        'series': 'Regularization Techniques',
        'level': 1,
        'description': 'Mixup data augmentation vs no-augmentation baseline on multi-class classification',
        'model_type': 'mlp_classifier',
        'loss_type': 'cross_entropy',
        'optimization': 'adam',
        'input_dim': 20,
        'output_dim': 6,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=64, val_ratio=0.2):
    """
    6-class classification with overlapping boundaries to make regularization
    clearly beneficial.
    """
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=12,
        n_redundant=4, n_classes=6, n_clusters_per_class=1,
        class_sep=0.8, flip_y=0.05, random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )

    mu, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim=20, num_classes=6, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(device=None, input_dim=20, num_classes=6):
    device = device or get_device()
    return ClassifierMLP(input_dim, num_classes).to(device)


def _mixup_data(x, y, alpha=0.4):
    """Apply Mixup augmentation and return mixed inputs, pair of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, val_loader, device=None, epochs=80, lr=1e-3, mixup_alpha=0.4):
    """Train with Mixup augmentation."""
    device = device or get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            mixed_x, y_a, y_b_mix, lam = _mixup_data(X_b, y_b, mixup_alpha)
            optimizer.zero_grad()
            out = model(mixed_x)
            loss = _mixup_criterion(criterion, out, y_a, y_b_mix, lam)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1

        loss_hist.append(running / n)
        val_m = evaluate(model, val_loader, device)
        val_loss_hist.append(val_m['loss'])

        if (epoch + 1) % 20 == 0:
            print(f"  [Mixup] Epoch {epoch+1}/{epochs}  train_loss={running/n:.4f}  "
                  f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}  "
                  f"val_f1={val_m['macro_f1']:.4f}")

    return {'loss_history': loss_hist, 'val_loss_history': val_loss_hist}


def _train_no_mixup(model, train_loader, val_loader, device, epochs, lr):
    """Baseline training without Mixup for comparison."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_hist, val_loss_hist = [], []

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
            running += loss.item()
            n += 1

        loss_hist.append(running / n)
        val_m = evaluate(model, val_loader, device)
        val_loss_hist.append(val_m['loss'])

        if (epoch + 1) % 20 == 0:
            print(f"  [NoMix] Epoch {epoch+1}/{epochs}  train_loss={running/n:.4f}  "
                  f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}  "
                  f"val_f1={val_m['macro_f1']:.4f}")

    return {'loss_history': loss_hist, 'val_loss_history': val_loss_hist}


def evaluate(model, data_loader, device=None):
    device = device or get_device()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_tgts = [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item()
            n += 1
            all_preds.append(logits.argmax(dim=1).cpu())
            all_tgts.append(y_b.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_tgts).numpy()

    acc = (y_pred == y_true).mean()
    mf1 = f1_score(y_true, y_pred, average='macro')
    mse = np.mean((y_pred - y_true) ** 2).item()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot

    return {
        'loss': total_loss / max(n, 1),
        'accuracy': float(acc),
        'macro_f1': float(mf1),
        'mse': float(mse),
        'r2': float(r2),
    }


def predict(model, X, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return model(X.to(device)).argmax(dim=1).cpu().numpy()


def save_artifacts(model, metrics, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Mixup Data Augmentation vs Baseline")
    print("=" * 60)

    EPOCHS = 80
    LR = 1e-3
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(batch_size=64)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Classes: {len(np.unique(y_train))}")

    # --- Mixup training ---
    print("\n--- Training WITH Mixup (alpha=0.4) ---")
    set_seed(42)
    model_mixup = build_model(device)
    history_mixup = train(model_mixup, train_loader, val_loader, device,
                          epochs=EPOCHS, lr=LR, mixup_alpha=0.4)
    mix_val = evaluate(model_mixup, val_loader, device)

    # --- Baseline (no mixup) ---
    print("\n--- Training WITHOUT Mixup ---")
    set_seed(42)
    model_base = build_model(device)
    history_base = _train_no_mixup(model_base, train_loader, val_loader, device, EPOCHS, LR)
    base_val = evaluate(model_base, val_loader, device)

    print("\n" + "-" * 60)
    print("Comparison Summary")
    print("-" * 60)
    print(f"Mixup     -> val_acc: {mix_val['accuracy']:.4f}  macro_F1: {mix_val['macro_f1']:.4f}")
    print(f"Baseline  -> val_acc: {base_val['accuracy']:.4f}  macro_F1: {base_val['macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)

    exit_code = 0

    if mix_val['macro_f1'] <= 0.60:
        print(f"FAIL: Mixup macro-F1 {mix_val['macro_f1']:.4f} <= 0.60")
        exit_code = 1
    else:
        print(f"PASS: Mixup macro-F1 > 0.60: {mix_val['macro_f1']:.4f}")

    if mix_val['accuracy'] <= 0.60:
        print(f"FAIL: Mixup accuracy {mix_val['accuracy']:.4f} <= 0.60")
        exit_code = 1
    else:
        print(f"PASS: Mixup accuracy > 0.60: {mix_val['accuracy']:.4f}")

    mix_train = evaluate(model_mixup, train_loader, device)
    base_train = evaluate(model_base, train_loader, device)
    mix_gap = abs(mix_train['accuracy'] - mix_val['accuracy'])
    base_gap = abs(base_train['accuracy'] - base_val['accuracy'])
    if mix_gap <= base_gap:
        print(f"PASS: Mixup generalizes better (gap {mix_gap:.4f} <= baseline gap {base_gap:.4f})")
    else:
        print(f"INFO: Mixup gap {mix_gap:.4f} > baseline gap {base_gap:.4f}")

    final_loss = history_mixup['loss_history'][-1]
    init_loss = history_mixup['loss_history'][0]
    if final_loss >= init_loss:
        print(f"FAIL: Loss did not decrease")
        exit_code = 1
    else:
        print(f"PASS: Loss decreased ({init_loss:.4f} -> {final_loss:.4f})")

    print("\nSaving artifacts...")
    save_artifacts(model_mixup, {
        'mixup_val': mix_val,
        'baseline_val': base_val,
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
