#!/usr/bin/env python3
"""
E-Prop Izhikevich Multi MNIST — Grid Search Tuner
==================================================
Использует PyEPropIzhikevichMulti (per-synapse eligibility) из taas_checkmate.
"""

import numpy as np
import time
import sys
from itertools import product

import torchvision
import torchvision.transforms as T

from taas_checkmate import PyEPropIzhikevichMulti

# ─── Data loading ─────────────────────────────────────────

def load_mnist(root="/tmp/mnist_data"):
    """Load MNIST as float32 numpy arrays."""
    transform = T.Compose([T.ToTensor()])  # [0, 1]
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    X_train = np.stack([img.numpy().ravel() for img, _ in train_set]).astype(np.float32)
    y_train = np.array([lbl for _, lbl in train_set], dtype=np.int32)
    X_test  = np.stack([img.numpy().ravel() for img, _ in test_set]).astype(np.float32)
    y_test  = np.array([lbl for _, lbl in test_set], dtype=np.int32)
    return X_train, y_train, X_test, y_test


# ─── Poisson encoding ─────────────────────────────────────

class PoissonEncoder:
    """Rate-based Poisson spike generator."""
    def __init__(self, n_input: int, dt_sec: float = 0.001, rng=None):
        self.n = n_input
        self.dt = dt_sec
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rates = np.zeros(n_input, dtype=np.float32)

    def set_rates(self, rates_hz: np.ndarray):
        self.rates[:] = rates_hz

    def step(self, spikes_out: np.ndarray) -> int:
        """Generate Poisson spikes. Returns spike count."""
        probs = self.rates * self.dt
        probs = probs.clip(0.0, 1.0)
        spikes_out[:] = (self.rng.random(self.n) < probs).astype(np.int32)
        return int(spikes_out.sum())


# ─── E-Prop MNIST class ──────────────────────────────────

class EPropMNIST:
    """E-Prop Izhikevich Multi SNN for MNIST classification."""

    def __init__(self, ni=784, nh=256, no=10,
                 dt=1.0, tau_e=30.0,
                 lr_in=0.005, lr_out=0.02,
                 bias=8.0, poisson_max_hz=200.0,
                 n_steps=50, seed=42,
                 d=2.0, tau_surrogate=5.0):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.n_steps = n_steps
        self.poisson_max_hz = poisson_max_hz
        self.seed = seed

        rng = np.random.default_rng(seed)

        # E-Prop hidden layer: ni → nh, per-synapse eligibility
        # Defaults: d=2.0 (low adaptation), tau_surrogate=5.0 (wide surrogate gradient)
        self.eprop = PyEPropIzhikevichMulti(ni, nh, dt, tau_e)

        # Input weights [ni × nh] column-major
        # 10x init + 10x kernel = 100x effective scale for input current
        # Combined with low bias, input completely dominates dynamics
        scale = 10.0 / np.sqrt(ni)
        self.w_in = rng.uniform(-scale, scale, ni * nh).astype(np.float32)

        # Output readout weights [nh × no]
        self.w_out = rng.uniform(-0.1, 0.1, nh * no).astype(np.float32)

        # Biases
        self.bias_hidden = np.full(nh, bias, dtype=np.float32)
        self.bias_out = np.zeros(no, dtype=np.float32)

        # Hyperparams
        self.lr_in = lr_in
        self.lr_out = lr_out
        self.dt = dt

        # Buffers (reusable)
        self.z_pre = np.zeros(ni, dtype=np.int32)
        self.spikes = np.zeros(nh, dtype=np.int32)
        self.hidden_count = np.zeros(nh, dtype=np.int32)
        self.logits = np.zeros(no, dtype=np.float32)
        self.probs = np.zeros(no, dtype=np.float32)
        self.learning_signals = np.zeros(nh, dtype=np.float32)
        self.elig_buf = np.zeros(ni * nh, dtype=np.float32)

        # Poisson encoder
        self.poisson = PoissonEncoder(ni, dt_sec=dt / 1000.0, rng=rng)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Run E-Prop forward over n_steps. Returns logits."""
        # Rate → Poisson spikes
        rates = image * self.poisson_max_hz
        self.poisson.set_rates(rates)

        # Reset hidden state
        self.eprop.reset()
        self.hidden_count.fill(0)

        for _ in range(self.n_steps):
            # Poisson step
            self.poisson.step(self.z_pre)
            # E-Prop fused forward
            self.eprop.step(self.w_in, self.z_pre, self.bias_hidden, self.spikes)
            self.hidden_count += self.spikes

        # Readout: logits[k] = Σ_j w_out[j,k] * hidden_count[j]
        out = self.hidden_count.astype(np.float32)  # [nh]
        self.logits[:] = out @ self.w_out.reshape(self.nh, self.no) + self.bias_out

        # Numerically stable softmax
        logits = self.logits - self.logits.max()
        exp = np.exp(logits)
        self.probs[:] = exp / (exp.sum() + 1e-30)
        return self.logits

    def backward(self, target: int):
        """Compute learning signals and update weights.
        
        Rust reference:
          error[k] = one_hot[k] - probs[k]
          d_w_out[k,j] += LR_OUT * error[k] * hidden[j]
          L[j] = Σ_k w_fb[j*NO + k] * error[k]
          d_w_in[i,j] = η * L[j] * e[i,j]
        """
        # One-hot target
        one_hot = np.zeros(self.no, dtype=np.float32)
        one_hot[target] = 1.0

        # Error: one_hot - probs (как в Rust, правильный знак)
        error = one_hot - self.probs  # [no]

        # Readout update: w_out += lr * outer(hidden, error)
        hidden = self.hidden_count.astype(np.float32)
        self.w_out += self.lr_out * np.outer(hidden, error).ravel()
        self.bias_out += self.lr_out * error

        # Learning signal for hidden layer: feedback alignment
        # L_j = Σ_k w_fb[j, k] * error[k]
        fb_scale = 1.0 / np.sqrt(self.no)
        if not hasattr(self, '_w_fb'):
            rng = np.random.default_rng(self.seed + 1)
            # w_fb flat [no * nh], reshape(no, nh) → transpose → (nh, no)
            # L = w_fb_flat.reshape(no, nh).T @ error = [nh]
            self._w_fb = rng.uniform(-fb_scale, fb_scale, self.no * self.nh).astype(np.float32)
        self.learning_signals[:] = self._w_fb.reshape(self.no, self.nh).T @ error

        # E-Prop per-synapse update for w_in
        self.eprop.update_weights(self.w_in, self.learning_signals, self.lr_in)

    def train_step(self, image: np.ndarray, target: int) -> tuple:
        """Single training step. Returns (loss, acc)."""
        self.forward(image)
        loss = -np.log(self.probs[target] + 1e-30)
        acc = 1 if self.probs.argmax() == target else 0
        self.backward(target)
        return loss, acc

    def predict(self, image: np.ndarray) -> int:
        """Forward pass only (no training)."""
        self.forward(image)
        return self.probs.argmax()

    def get_firing_rate(self) -> float:
        """Average firing rate over last forward pass (spk/step)."""
        return float(self.hidden_count.sum()) / (self.nh * self.n_steps)


# ─── Training loop ────────────────────────────────────────

def train_epoch(model, X, y, batch_size=1, shuffle=True):
    """Train for one epoch. Shuffle is ignored for batch_size=1 cycling."""
    n = len(X)
    losses, accs = [], []
    frs = []

    if shuffle:
        idx = np.random.default_rng().permutation(n)
    else:
        idx = np.arange(n)

    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        for bi in batch_idx:
            loss, acc = model.train_step(X[bi], y[bi])
            losses.append(loss)
            accs.append(acc)
        # Report every 10K samples
        if (i // batch_size + 1) % 2000 == 0:
            avg_loss = np.mean(losses[-2000:]) if len(losses) >= 2000 else np.mean(losses)
            avg_acc = np.mean(accs[-2000:]) if len(accs) >= 2000 else np.mean(accs)
            fr = model.get_firing_rate()
            frs.append(fr)
            print(f"  [{i//batch_size:6d}/{n//batch_size:6d}]  loss={avg_loss:.4f}  acc={avg_acc:.3f}  FR={fr:.4f}")

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)
    avg_fr = np.mean(frs) if frs else model.get_firing_rate()
    return avg_loss, avg_acc, avg_fr


def evaluate(model, X, y, max_samples=None):
    """Evaluate model on dataset."""
    n = len(X) if max_samples is None else min(len(X), max_samples)
    correct = 0
    for i in range(n):
        pred = model.predict(X[i])
        correct += (pred == y[i])
        if (i + 1) % 5000 == 0:
            print(f"    Eval [{i+1}/{n}]  acc={correct/(i+1):.4f}")
    return correct / n


# ─── Grid search ──────────────────────────────────────────

def run_trial(params, X_train, y_train, X_test, y_test, n_train=10000):
    """Run one training trial with given hyperparams."""
    print(f"\n{'='*60}")
    print(f"Trial: {params}")
    print(f"{'='*60}")

    model = EPropMNIST(
        ni=784, nh=params['nh'], no=10,
        dt=params['dt'],
        tau_e=params['tau_e'],
        lr_in=params['lr_in'],
        lr_out=params['lr_out'],
        bias=params['bias'],
        poisson_max_hz=params['poisson_max_hz'],
        n_steps=params['n_steps'],
        seed=params.get('seed', 42),
    )

    t0 = time.time()

    # Train on subset
    print(f"Training on {n_train} samples...")
    train_loss, train_acc, train_fr = train_epoch(model, X_train[:n_train], y_train[:n_train])

    t_train = time.time() - t0

    # Test
    print("Evaluating on test set...")
    test_acc = evaluate(model, X_test, y_test)

    t_total = time.time() - t0
    img_s = n_train / t_train

    result = {
        **params,
        'train_loss': round(train_loss, 4),
        'train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'firing_rate': round(train_fr, 4),
        'img_s': round(img_s, 1),
        'time_s': round(t_total, 1),
    }

    print(f"\n  → train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"test_acc={test_acc:.4f}  FR={train_fr:.4f}  {img_s:.0f} img/s\n")

    return result


# ─── Main ─────────────────────────────────────────────────

def main():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Default params — 10x kernel + 10x init + larger bias
    # bias=8 gives higher FR; weight signal still dominates (std~26 vs bias=8)
    default_params = {
        'nh': 256,
        'dt': 1.0,
        'tau_e': 30.0,
        'lr_in': 0.05,
        'lr_out': 0.02,
        'bias': 8.0,
        'poisson_max_hz': 200.0,
        'n_steps': 50,
        'seed': 42,
    }

    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        # ─── Grid search ────────────────────────────
        grid = {
            'lr_in':  [0.0005, 0.001, 0.002, 0.005],
            'lr_out': [0.005, 0.01, 0.02],
            'bias':   [3.0, 4.0, 5.0, 6.0],
            'poisson_max_hz': [100, 200, 400],
        }

        keys = list(grid.keys())
        results = []

        for values in product(*grid.values()):
            params = default_params.copy()
            for k, v in zip(keys, values):
                params[k] = v

            result = run_trial(params, X_train, y_train, X_test, y_test, n_train=10000)
            results.append(result)

        # Sort by test accuracy
        results.sort(key=lambda r: r['test_acc'], reverse=True)

        print("\n" + "="*70)
        print("TOP 5 RESULTS")
        print("="*70)
        print(f"{'Rank':<5} {'test_acc':<10} {'train_acc':<10} {'FR':<8} {'lr_in':<8} {'lr_out':<8} {'bias':<8} {'max_hz':<8}")
        print("-"*70)
        for i, r in enumerate(results[:5]):
            print(f"{i+1:<5} {r['test_acc']:<10.4f} {r['train_acc']:<10.4f} {r['firing_rate']:<8.4f} "
                  f"{r['lr_in']:<8} {r['lr_out']:<8} {r['bias']:<8} {r['poisson_max_hz']:<8}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'fast':
        # ─── Fast test ──────────────────────────────
        params = default_params.copy()
        params['nh'] = 128
        params['n_steps'] = 15
        result = run_trial(params, X_train, y_train, X_test, y_test, n_train=3000)
        print(f"Fast result: {result}")

    else:
        # ─── Full training ──────────────────────────
        # Just run with default params on more data
        params = default_params.copy()
        print(f"\nTraining with params: {params}")
        result = run_trial(params, X_train, y_train, X_test, y_test, n_train=60000)
        print(f"Final: {result}")


if __name__ == '__main__':
    main()
