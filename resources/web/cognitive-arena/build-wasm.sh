#!/usr/bin/env bash
# Build the Rust-owned Cognitive Arena core to WebAssembly for the web UI.
#
# This is the single reproducible entry point referenced by README.md and the
# browser UI. The generated `pkg/` directory is build output and is git-ignored.
#
# Prerequisites (once):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-pack
set -euo pipefail

ui_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${ui_dir}/../../.." && pwd)"
out="${ui_dir}/pkg"

echo ">> wasm-pack build (root=${root})"
# wasm-pack 0.15 runs cargo from the current directory and writes `pkg/`
# there by default; its `--artifact-dir` flag requires nightly cargo, so we
# build in the repo root and move the result into the UI directory.
(cd "${root}" && wasm-pack build \
  --target web \
  --no-typescript \
  --no-default-features \
  --features wasm)
rm -rf "${out}"
mv "${root}/pkg" "${out}"

echo ">> built: ${out}"
echo ">> serve: python -m http.server 8080 -d ${ui_dir}"
