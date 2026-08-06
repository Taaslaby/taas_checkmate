#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Step 1: Compile NEURON .mod files ==="
# Try to compile with nrnivmodl if available
if command -v nrnivmodl &> /dev/null; then
    nrnivmodl *.mod
    echo "✅ Compilation done"
else
    echo "⚠️  nrnivmodl not found, checking for pre-compiled lib..."
fi

echo ""
echo "=== Step 2: Run validation ==="
cd ../..
PYTHON=.venv/bin/python3 validate/neuron_models/compare_all.py "$@"
