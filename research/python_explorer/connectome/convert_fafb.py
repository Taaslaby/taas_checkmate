#!/usr/bin/env python3
"""
Convert FAFB connectome CSV → CSR NPZ for benchmark.

Input:
  /home/taaslab/Загрузки/connections_princeton_no_threshold.csv.gz  (22.3M synapses)
  /home/taaslab/Загрузки/neurons(1).csv.gz                          (139K neurons)

Output:
  /home/taaslab/Проекты/t3-wavelet/data/connectome/fafb_139k.npz
"""

import gzip, csv, os, sys, time
import numpy as np

LOAD_DIR = "/home/taaslab/Загрузки"
OUT_DIR = "/home/taaslab/Проекты/t3-wavelet/data/connectome"
NEURON_FILE = os.path.join(LOAD_DIR, "neurons(1).csv.gz")
CONN_FILE = os.path.join(LOAD_DIR, "connections_princeton_no_threshold.csv.gz")
OUT_PATH = os.path.join(OUT_DIR, "fafb_139k.npz")


def load_neurons(path):
    """Load root_ids from neurons CSV, return dict root_id → idx."""
    print(f"  Loading neurons from {path}...", end=" ", flush=True)
    root_ids = []
    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            root_ids.append(int(row["root_id"]))
    print(f"✅ {len(root_ids):,} neurons")
    # Build reverse mapping
    print("  Building index mapping...", end=" ", flush=True)
    id_to_idx = {rid: i for i, rid in enumerate(root_ids)}
    print(f"✅")
    return id_to_idx


def build_csr(id_to_idx, conn_path):
    """Build CSR from connections CSV using the ID mapping."""
    n_neurons = len(id_to_idx)

    # First pass: count synapses per pre-neuron
    print(f"  First pass: counting synapses...", end=" ", flush=True)
    pre_counts = np.zeros(n_neurons, dtype=np.int32)
    total_valid = 0
    total_skipped = 0
    with gzip.open(conn_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = int(row["pre_root_id"])
            post = int(row["post_root_id"])
            pre_idx = id_to_idx.get(pre)
            post_idx = id_to_idx.get(post)
            if pre_idx is not None and post_idx is not None:
                pre_counts[pre_idx] += 1
                total_valid += 1
            else:
                total_skipped += 1
                if total_skipped <= 3:
                    print(f"\n    Skipped: pre={pre} ({pre_idx}), post={post} ({post_idx})")
    print(f"✅ {total_valid:,} valid, {total_skipped:,} skipped")

    # Build row_ptr from counts
    row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
    np.cumsum(pre_counts, out=row_ptr[1:])

    # Allocate col_indices and values
    col_indices = np.zeros(total_valid, dtype=np.int32)
    values = np.zeros(total_valid, dtype=np.float32)
    # Per-neuron cursor
    cursor = row_ptr[1:].copy() - pre_counts  # start positions for each neuron

    # Second pass: fill CSR data
    print(f"  Second pass: filling CSR data...", end=" ", flush=True)
    with gzip.open(conn_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = int(row["pre_root_id"])
            post = int(row["post_root_id"])
            syn_count = int(row["syn_count"])
            pre_idx = id_to_idx.get(pre)
            post_idx = id_to_idx.get(post)
            if pre_idx is not None and post_idx is not None:
                pos = cursor[pre_idx]
                col_indices[pos] = post_idx
                values[pos] = syn_count  # synaptic weight = synapse count
                cursor[pre_idx] = pos + 1
    print(f"✅")

    print(f"  Stats: row_ptr={len(row_ptr)}, col_indices={len(col_indices)}, values={len(values)}")
    print(f"  Neurons: {n_neurons:,}, Synapses: {total_valid:,}")
    print(f"  Avg fan-out: {total_valid / n_neurons:.1f}")

    return n_neurons, row_ptr, col_indices, values


def verify_csr(n, row_ptr, col_indices, values):
    """Quick sanity check."""
    print("  Verifying CSR...", end=" ", flush=True)
    assert len(row_ptr) == n + 1
    assert len(col_indices) == len(values)
    assert row_ptr[0] == 0
    assert row_ptr[-1] == len(values)
    # Check all indices are in range
    assert col_indices.min() >= 0
    assert col_indices.max() < n
    # Check row_ptr is non-decreasing
    assert np.all(row_ptr[1:] >= row_ptr[:-1])
    print(f"✅ row_ptr[-1]={row_ptr[-1]:,}, col_indices in [0, {n}), OK")


if __name__ == "__main__":
    print("=" * 82)
    print("  FAFB Connectome Converter: CSV → CSR NPZ")
    print("=" * 82)
    print()

    t0 = time.perf_counter()

    # Load neurons
    id_to_idx = load_neurons(NEURON_FILE)
    n_neurons = len(id_to_idx)

    # Build CSR
    n, row_ptr, col_indices, values = build_csr(id_to_idx, CONN_FILE)

    # Verify
    verify_csr(n, row_ptr, col_indices, values)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  Saving to {OUT_PATH}...", end=" ", flush=True)
    np.savez_compressed(OUT_PATH,
        values=values,
        col_indices=col_indices,
        row_ptr=row_ptr,
    )
    out_size = os.path.getsize(OUT_PATH)
    print(f"✅ ({out_size / 1e6:.1f} MB)")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {elapsed:.1f}s")
    print()
