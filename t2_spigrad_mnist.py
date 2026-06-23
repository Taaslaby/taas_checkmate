#!/usr/bin/env python3
"""
T2-SPIGRAD on MNIST — spike train classification through differentiable T3 dynamics.

Compares activations: ReLU, GELU, SiLU, LeakyReLU
Architecture:  SNN (Linear+Act+LN+Drop → Linear+Act+LN+Drop) → T3 (n_osc=128, n_steps=100) → PST → classifier
Baselines:     Pure SNN (2-layer MLP), Pure T3 (frozen random SNN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
from taas_checkmate import t2spigrad_forward, t2spigrad_backward, pst_linear_dim


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

N_OSC = 16
N_STEPS = 20
DT = 0.05
DRIVE_GAIN = 10.0
ALPHA = 1.0
FREQ_MIN = 0.5
FREQ_MAX = 3.0
COUPLING = (0.3, 0.15, 0.08)

INPUT_SIZE = 784
HIDDEN_SIZE = 3 * N_OSC  # 48
PST_DIM = pst_linear_dim(N_OSC)  # = 3*16*6 + 12 = 300
NUM_CLASSES = 10

N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 16
N_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"T3: n_osc={N_OSC}, n_steps={N_STEPS}, pst_dim={PST_DIM}")
print(f"Data: {N_TRAIN} train, {N_TEST} test, batch={BATCH_SIZE}")


# ═══════════════════════════════════════════════════════════════
#  Activation factory
# ═══════════════════════════════════════════════════════════════

def get_activation(name):
    """Return activation module by name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU(approximate='tanh')
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'leaky_relu' or name == 'lrelu':
        return nn.LeakyReLU(0.1)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError(f"Unknown activation: {name} (try: relu, gelu, silu, leaky_relu)")


# ═══════════════════════════════════════════════════════════════
#  1. MNIST → Poisson rate coding
# ═══════════════════════════════════════════════════════════════

def load_mnist(n_train=N_TRAIN, n_test=N_TEST, seed=42):
    """Load MNIST and generate Poisson spike trains."""
    print("\n  Loading MNIST...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root="/tmp/mnist_data", train=True,
                                    download=True, transform=transform)
        test_set = datasets.MNIST(root="/tmp/mnist_data", train=False,
                                   download=True, transform=transform)
    except ImportError:
        print("  torchvision not found, generating synthetic data")
        return generate_synthetic_mnist(n_train, n_test)

    rng = np.random.default_rng(seed)

    def images_to_spikes(dataset, n_samples):
        spike_trains, y = [], []
        for idx in range(n_samples):
            img_tensor, label = dataset[idx]
            pixels = img_tensor.numpy().flatten().astype(np.float32)
            spikes = (rng.random((N_STEPS, INPUT_SIZE)) < pixels).astype(np.float32)
            spike_trains.append(spikes)
            y.append(label)
        return np.array(spike_trains), np.array(y, dtype=np.int64)

    X_train_spikes, y_train = images_to_spikes(train_set, n_train)
    X_test_spikes, y_test = images_to_spikes(test_set, n_test)

    print(f"  Train: {X_train_spikes.shape}  Test: {X_test_spikes.shape}")
    print(f"  Mean spike rate: {X_train_spikes.mean():.3f}")
    return X_train_spikes, y_train, X_test_spikes, y_test


def generate_synthetic_mnist(n_train, n_test):
    rng = np.random.default_rng(42)
    X_train = (rng.random((n_train, N_STEPS, INPUT_SIZE)) < 0.3).astype(np.float32)
    y_train = rng.integers(0, NUM_CLASSES, n_train)
    X_test = (rng.random((n_test, N_STEPS, INPUT_SIZE)) < 0.3).astype(np.float32)
    y_test = rng.integers(0, NUM_CLASSES, n_test)
    return X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════
#  2. T2-SPIGRAD PyTorch autograd Function
# ═══════════════════════════════════════════════════════════════

class T2SpigradFunction(torch.autograd.Function):
    """Differentiable T3 encoder — forward + backward через Rust."""

    @staticmethod
    def forward(ctx, drives, n_osc, n_steps, dt,
                drive_gain=DRIVE_GAIN, alpha=ALPHA,
                freq_min=FREQ_MIN, freq_max=FREQ_MAX,
                coupling=COUPLING):
        batch_size, seq_len, n_total = drives.shape
        drives_np = drives.detach().cpu().numpy()
        all_pst, all_trajs = [], []
        for b in range(batch_size):
            d = np.ascontiguousarray(drives_np[b])
            pst_arr, traj = t2spigrad_forward(
                n_osc, n_steps, dt, drive_gain, alpha,
                freq_min, freq_max,
                coupling[0], coupling[1], coupling[2], d,
            )
            all_pst.append(pst_arr)
            all_trajs.append(traj)
        ctx.trajs = all_trajs
        pst_out = np.stack(all_pst, axis=0)
        return torch.tensor(pst_out, device=drives.device, dtype=drives.dtype)

    @staticmethod
    def backward(ctx, grad_pst):
        batch_size = grad_pst.shape[0]
        device = grad_pst.device
        grad_drives_list = []
        for b in range(batch_size):
            gp = grad_pst[b].cpu().contiguous().numpy()
            gd = t2spigrad_backward(gp, ctx.trajs[b])
            grad_drives_list.append(np.ascontiguousarray(gd))
        grad_drives = np.stack(grad_drives_list, axis=0)
        result = torch.as_tensor(grad_drives, device=device, dtype=grad_pst.dtype)
        return result, None, None, None, None, None, None, None, None, None


# ═══════════════════════════════════════════════════════════════
#  3. T2-SPIGRAD Full Model (configurable activation)
# ═══════════════════════════════════════════════════════════════

class T2SpigradModel(nn.Module):
    """SNN (Linear+Act+LN+Drop → Linear+Act+LN+Drop) → T3 → classifier."""

    def __init__(self, activation='gelu', dropout=DROPOUT):
        super().__init__()
        act = get_activation(activation)
        self.snn = nn.Sequential(
            nn.Linear(INPUT_SIZE, INPUT_SIZE // 2),          # 784→392
            nn.LayerNorm(INPUT_SIZE // 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(INPUT_SIZE // 2, HIDDEN_SIZE),         # 392→384
            nn.LayerNorm(HIDDEN_SIZE),
            act,
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(PST_DIM, NUM_CLASSES)

    def forward(self, spikes):
        drives = self.snn(spikes)
        pst = T2SpigradFunction.apply(drives, N_OSC, N_STEPS, DT)
        return self.classifier(pst)


# ═══════════════════════════════════════════════════════════════
#  4. Pure SNN Baseline
# ═══════════════════════════════════════════════════════════════

class PureSNN(nn.Module):
    """Standard 2-layer NN: 784 → FC(384, ReLU) → FC(10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, spikes):
        x = spikes.mean(dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ═══════════════════════════════════════════════════════════════
#  5. Pure T3 Baseline (frozen random SNN)
# ═══════════════════════════════════════════════════════════════

class PureT3Model(nn.Module):
    """Random SNN (frozen) → T3 → trainable classifier."""

    def __init__(self):
        super().__init__()
        self.snn = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)
        nn.init.normal_(self.snn.weight, std=0.5)
        for p in self.snn.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(PST_DIM, NUM_CLASSES)

    def forward(self, spikes):
        with torch.no_grad():
            drives = self.snn(spikes)
        pst = T2SpigradFunction.apply(drives, N_OSC, N_STEPS, DT).detach()
        return self.classifier(pst)


# ═══════════════════════════════════════════════════════════════
#  6. Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for spikes, labels in loader:
        spikes, labels = spikes.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(spikes)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * spikes.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += spikes.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for spikes, labels in loader:
        spikes, labels = spikes.to(device), labels.to(device)
        logits = model(spikes)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * spikes.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += spikes.size(0)
    return total_loss / total, correct / total


def run_training(name, model_class, X_train, y_train, X_test, y_test,
                 model_kwargs=None):
    """Create, train, and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # Warmup + benchmark
    sample_spikes, _ = next(iter(train_loader))
    sample_spikes = sample_spikes.to(DEVICE)
    _ = model(sample_spikes)  # warmup (PyO3 init)
    t0 = time.time()
    _ = model(sample_spikes)
    t_batch = time.time() - t0
    print(f"  Batch fwd: {t_batch:.3f}s")

    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Test Loss':>10}  {'Test Acc':>10}  {'Time':>8}")
    print(f"  {'-'*50}")

    best_test_acc = 0.0
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)
        scheduler.step()
        dt = time.time() - t0
        best_test_acc = max(best_test_acc, test_acc)
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {train_acc:>10.2%}  "
              f"{test_loss:>10.4f}  {test_acc:>10.2%}  {dt:>7.1f}s")

    print(f"\n  Best test acc: {best_test_acc:.2%}")
    return best_test_acc, model


# ═══════════════════════════════════════════════════════════════
#  7. Activation comparison
# ═══════════════════════════════════════════════════════════════

ACTIVATIONS = ['relu', 'gelu', 'silu', 'leaky_relu']


def run_activation_comparison(X_train, y_train, X_test, y_test):
    """Train T2-SPIGRAD with each activation and report results."""
    print(f"\n{'='*60}")
    print("  ACTIVATION COMPARISON — T2-SPIGRAD")
    print(f"{'='*60}")

    results = {}
    for act_name in ACTIVATIONS:
        acc, model = run_training(
            f"T2-SPIGRAD ({act_name})", T2SpigradModel,
            X_train, y_train, X_test, y_test,
            model_kwargs={'activation': act_name},
        )
        results[f"T2-SPIGRAD ({act_name})"] = acc

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  T2-SPIGRAD — MNIST Activation Comparison")
    print(f"  n_osc={N_OSC}, n_steps={N_STEPS}, {N_EPOCHS} epochs")
    print("═" * 60)

    # 1. Load data
    X_train, y_train, X_test, y_test = load_mnist()

    # 2. Activation comparison
    results = {}
    for act_name in ACTIVATIONS:
        acc, _ = run_training(
            f"T2-SPIGRAD ({act_name})", T2SpigradModel,
            X_train, y_train, X_test, y_test,
            model_kwargs={'activation': act_name},
        )
        results[f"T2-SPIGRAD ({act_name})"] = acc

    # 3. Baselines
    acc, _ = run_training("Pure SNN", PureSNN,
                          X_train, y_train, X_test, y_test)
    results["Pure SNN"] = acc

    acc, _ = run_training("Pure T3 (frozen)", PureT3Model,
                          X_train, y_train, X_test, y_test)
    results["Pure T3"] = acc

    # 4. Final comparison
    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Model':<30}  {'Test Acc':>10}")
    print(f"  {'-'*42}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = "  ← BEST" if acc == max(results.values()) else ""
        print(f"  {name:<30}  {acc:>8.2%}{marker}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

