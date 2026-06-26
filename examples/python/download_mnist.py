#!/usr/bin/env python3
"""Download MNIST from alternative mirror using only stdlib."""
import urllib.request
import gzip
import os
import sys

DATA_DIR = "examples/data"
os.makedirs(DATA_DIR, exist_ok=True)

# TensorFlow CDN mirror (has the same IDX.gz files)
MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist"

FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
}

# Also try alternative mirrors
MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
]

for mirror_url in MIRRORS:
    print(f"Trying mirror: {mirror_url}")
    success = True
    for fname, extracted_name in FILES.items():
        url = f"{mirror_url}/{fname}"
        gz_path = os.path.join(DATA_DIR, fname)
        out_path = os.path.join(DATA_DIR, extracted_name)

        if os.path.exists(out_path):
            print(f"  ✓ {extracted_name} already exists")
            continue

        print(f"  ⬇ {url}")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"    → saved {gz_path}")

            # Decompress
            with gzip.open(gz_path, 'rb') as f_in:
                data = f_in.read()
            with open(out_path, 'wb') as f_out:
                f_out.write(data)
            print(f"    → extracted {out_path} ({len(data)} bytes)")
            os.remove(gz_path)
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            success = False
            break

    if success:
        print(f"\n✓ All files downloaded from {mirror_url}")
        sys.exit(0)

print("\n✗ All mirrors failed")
sys.exit(1)
