#!/usr/bin/env python3
"""
Validate auto-tuner: train MNIST with config from suggest_eprop_config,
compare accuracy with empirical baseline (72.8% with bias=8).
"""

import numpy as np
import time
import sys

import torchvision
import torchvision.transforms as T

from taas_checkmate import PyEPropIzhikevichMulti, suggest_eprop_config

# ─── Reproduce tune_eprop_mnist.py classes ───────────────

def load_mnist(root="/tmp/mnist_data"):
    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    X_train = np.stack([img.numpy().ravel() for img, _ in train_set]).astype(np.float32)
    y_train = np.array([lbl for _, lbl in train_set], dtype=np.int32)
    X_test  = np.stack([img.numpy().ravel() for img, _ in test_set]).astype(np.float32)
    y_test  = np.array([lbl for _, lbl in test_set], dtype=np.int32)
    return X_train, y_train, X_test, y_test


class PoissonEncoder:
    def __init__(self, n_input, dt_sec=0.001, rng=None):
        self.n = n_input
        self.dt = dt_sec
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rates = np.zeros(n_input, dtype=np.float32)

    def set_rates(self, rates_hz):
        self.rates[:] = rates_hz

    def step(self, spikes_out):
        probs = (self.rates * self.dt).clip(0.0, 1.0)
        spikes_out[:] = (self.rng.random(self.n) < probs).astype(np.int32)
        return int(spikes_out.sum())


class EPropMNIST:
    """E-Prop Izhikevich Multi SNN for MNIST — tuner-driven init."""

    def __init__(self, config: dict, seed=42):
        """
        config: dict from suggest_eprop_config()
        """
        self.ni = 784
        self.nh = 256
        self.no = 10
        self.n_steps = config['n_steps']
        self.poisson_max_hz = 200.0
        self.seed = seed

        rng = np.random.default_rng(seed)

        # E-Prop with tau_e from config (default 30)
        self.eprop = PyEPropIzhikevichMulti(self.ni, self.nh, dt=1.0, tau_e=config['tau_e'])

        # Weight init: use weight_scale from tuner
        scale = config['weight_scale']
        self.w_in = rng.uniform(-scale, scale, self.ni * self.nh).astype(np.float32)

        # Output readout
        self.w_out = rng.uniform(-0.1, 0.1, self.nh * self.no).astype(np.float32)

        # Bias from tuner
        self.bias_hidden = np.full(self.nh, config['bias'], dtype=np.float32)
        self.bias_out = np.zeros(self.no, dtype=np.float32)

        # Hyperparams from tuner
        self.lr_in = config['lr_in']
        self.lr_out = config['lr_out']

        # Buffers
        self.z_pre = np.zeros(self.ni, dtype=np.int32)
        self.spikes = np.zeros(self.nh, dtype=np.int32)
        self.hidden_count = np.zeros(self.nh, dtype=np.int32)
        self.logits = np.zeros(self.no, dtype=np.float32)
        self.probs = np.zeros(self.no, dtype=np.float32)
        self.learning_signals = np.zeros(self.nh, dtype=np.float32)

        self.poisson = PoissonEncoder(self.ni, dt_sec=1.0 / 1000.0, rng=rng)

    def forward(self, image):
        rates = image * self.poisson_max_hz
        self.poisson.set_rates(rates)
        self.eprop.reset()
        self.hidden_count.fill(0)

        for _ in range(self.n_steps):
            self.poisson.step(self.z_pre)
            self.eprop.step(self.w_in, self.z_pre, self.bias_hidden, self.spikes)
            self.hidden_count += self.spikes

        out = self.hidden_count.astype(np.float32)
        self.logits[:] = out @ self.w_out.reshape(self.nh, self.no) + self.bias_out
        logits = self.logits - self.logits.max()
        exp = np.exp(logits)
        self.probs[:] = exp / (exp.sum() + 1e-30)
        return self.logits

    def backward(self, target):
        one_hot = np.zeros(self.no, dtype=np.float32)
        one_hot[target] = 1.0
        error = one_hot - self.probs

        hidden = self.hidden_count.astype(np.float32)
        self.w_out += self.lr_out * np.outer(hidden, error).ravel()
        self.bias_out += self.lr_out * error

        fb_scale = 1.0 / np.sqrt(self.no)
        if not hasattr(self, '_w_fb'):
            rng = np.random.default_rng(self.seed + 1)
            self._w_fb = rng.uniform(-fb_scale, fb_scale, self.no * self.nh).astype(np.float32)
        self.learning_signals[:] = self._w_fb.reshape(self.no, self.nh).T @ error

        self.eprop.update_weights(self.w_in, self.learning_signals, self.lr_in)

    def train_step(self, image, target):
        self.forward(image)
        loss = -np.log(self.probs[target] + 1e-30)
        acc = 1 if self.probs.argmax() == target else 0
        self.backward(target)
        return loss, acc

    def predict(self, image):
        self.forward(image)
        return self.probs.argmax()

    def get_firing_rate(self):
        return float(self.hidden_count.sum()) / (self.nh * self.n_steps)


def train_epoch(model, X, y, n_train, report_every=2000):
    n = min(len(X), n_train)
    idx = np.random.default_rng().permutation(n)
    losses, accs, frs = [], [], []

    for i in range(n):
        loss, acc = model.train_step(X[idx[i]], y[idx[i]])
        losses.append(loss)
        accs.append(acc)
        if (i + 1) % report_every == 0:
            avg_loss = np.mean(losses[-report_every:])
            avg_acc = np.mean(accs[-report_every:])
            fr = model.get_firing_rate()
            frs.append(fr)
            print(f"  [{i+1:6d}/{n:6d}]  loss={avg_loss:.4f}  acc={avg_acc:.3f}  FR={fr:.4f}")

    return np.mean(losses), np.mean(accs), np.mean(frs) if frs else model.get_firing_rate()


def evaluate(model, X, y, max_samples=10000):
    n = min(len(X), max_samples)
    correct = 0
    for i in range(n):
        pred = model.predict(X[i])
        correct += (pred == y[i])
        if (i + 1) % 5000 == 0:
            print(f"    Eval [{i+1}/{n}]  acc={correct/(i+1):.4f}")
    return correct / n


# ─── Main ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Auto-Tuner Validation — MNIST E-Prop Izhikevich")
    print("=" * 60)

    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Get config from auto-tuner
    print("\nGetting config from suggest_eprop_config(784, 256, 0.2, 0.1, 42)...")
    tuner_config = suggest_eprop_config(784, 256, 0.2, 0.1, 42)
    for k, v in tuner_config.items():
        print(f"  {k}: {v}")

    # How many training samples
    n_train = 60000 if len(sys.argv) < 2 else int(sys.argv[1])

    print(f"\n{'='*60}")
    print(f"Training with tuner config on {n_train} samples")
    print(f"{'='*60}")

    model = EPropMNIST(tuner_config, seed=42)

    t0 = time.time()
    train_loss, train_acc, train_fr = train_epoch(model, X_train, y_train, n_train)
    train_time = time.time() - t0

    print(f"\nEvaluating on test set...")
    test_acc = evaluate(model, X_test, y_test)
    total_time = time.time() - t0
    img_s = n_train / train_time

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Tuner config:")
    print(f"    bias={tuner_config['bias']}, weight_scale={tuner_config['weight_scale']:.4f}")
    print(f"    lr_in={tuner_config['lr_in']}, lr_out={tuner_config['lr_out']}")
    print(f"    snr={tuner_config['snr']:.2f}")
    print(f"  Training: {n_train} samples, {total_time:.1f}s ({img_s:.0f} img/s)")
    print(f"  Train loss={train_loss:.4f}, train acc={train_acc:.4f}")
    print(f"  Test acc={test_acc:.4f}")
    print(f"  FR={train_fr:.4f}")
    print()
    print(f"  Baseline (manual tune, bias=8): 72.80% test acc")
    print(f"  Tuner (suggested bias=3):       {test_acc*100:.2f}% test acc")
    diff = test_acc * 100 - 72.80
    print(f"  Difference: {diff:+.2f} pp")
    print()


if __name__ == '__main__':
    main()
