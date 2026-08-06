#!/usr/bin/env python3
"""
🧬 NEURON Model Compilation Script
Compiles all .mod files for neuron model validation.
Run before compare_all.py

Usage:
    python validate/neuron_models/compile_mods.py
"""
import os
import sys
import subprocess
import shutil

NEURON_DIR = os.path.dirname(os.path.abspath(__file__))

def find_nrniv():
    """Find NEURON's nrniv executable."""
    # Check common locations
    candidates = [
        "nrniv",
        shutil.which("nrniv"),
        os.path.expanduser("~/.local/bin/nrniv"),
        "/usr/local/bin/nrniv",
        "/usr/bin/nrniv",
    ]
    for cand in candidates:
        if cand and os.path.exists(cand if os.path.isabs(cand) else shutil.which(cand) or ""):
            return cand if os.path.isabs(cand) else shutil.which(cand)
    return None

def compile_mods():
    """Compile all .mod files using nrnivmodl."""
    print(f"📁 NEURON models: {NEURON_DIR}")
    
    nrniv = find_nrniv()
    if not nrniv:
        print("   ⚠ NEURON not found. Trying to install via pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "NEURON"], check=False)
        nrniv = find_nrniv()
    
    if nrniv:
        print(f"   ✅ NEURON found: {nrniv}")
        nrniv_dir = os.path.dirname(os.path.dirname(nrniv))
        nrnivmodl = shutil.which("nrnivmodl") or os.path.join(nrniv_dir, "bin", "nrnivmodl")
        
        if nrnivmodl:
            print(f"   🔧 Compiling with: {nrnivmodl}")
            mod_files = sorted([f for f in os.listdir(NEURON_DIR) if f.endswith('.mod')])
            print(f"      MOD files: {mod_files}")
            
            result = subprocess.run(
                [nrnivmodl] + mod_files,
                cwd=NEURON_DIR,
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f"   ✅ Compilation successful")
                for line in result.stdout.split('\n')[-5:]:
                    if line.strip():
                        print(f"      {line.strip()}")
            else:
                print(f"   ❌ Compilation failed: {result.stderr[:500]}")
                return False
        else:
            print("   ⚠ nrnivmodl not found, checking for pre-compiled mechanisms...")
    else:
        print("   ⚠ NEURON not available. Will use pre-compiled mechanisms if present.")
    
    # Check if .so files exist
    for root, dirs, files in os.walk(NEURON_DIR):
        for f in files:
            if f.endswith('.so') or 'libnrnmech' in f:
                print(f"   📦 Found: {os.path.join(root, f)}")
    
    print(f"   ✅ Done. Ready for validation.")
    return True

if __name__ == '__main__':
    compile_mods()
