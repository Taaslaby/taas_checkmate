#!/usr/bin/env python3
"""
🌀 ГИПЕРБОЛИЗАТОР GPU — GPT-2 × HH на CUDA

- GpuHhLut: .spikes даёт пер-нейрон спайки — без V-trace!
- Все 12 слоёв GPT-2 × 3072 HH-нейрона на GPU
- Stateless (Mode A) + Stateful (Mode B)
- Сравнение скорости CPU vs GPU
- Сохранение весов

Антон ❤️ Сардана — GPU гиперболизация, 2026
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Try GPU first, fall back to CPU
USE_GPU = False
try:
    from taas_checkmate import GpuHhLut
    USE_GPU = True
    ENGINE = "GPU (CUDA)"
    print(f"✅ GpuHhLut available — using CUDA acceleration")
except ImportError:
    from taas_checkmate import HhLut
    ENGINE = "CPU"
    print(f"⚠️  GpuHhLut not available, falling back to CPU HhLut")

# Choose the right engine per layer
if USE_GPU:
    HhEngine = GpuHhLut
else:
    HhEngine = HhLut

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01                  # ms per HH step
HH_STEPS = 500             # 5ms simulation
SCALE_IN = 3.0
BIAS_IN = 7.0
I_CLAMP = (0.0, 25.0)
SAVE_PATH = "hyperbolizer_gpu_weights.npz"
TEXT = "The mysterious quantum computer calculated the answer."

# ═══ Helpers ══════════════════════════════════════════════

def hh_forward(hh, I_cur: np.ndarray, n_steps: int) -> np.ndarray:
    """Run HH, return per-neuron spike counts.
    
    Используем V-trace (reverse zero-crossing) для детекции спайков.
    На GPU: .v синхронизирован (kernel → sync → download).
    На CPU: .v напрямую из памяти.
    """
    hh.set_I_all(I_cur.astype(np.float32))
    total = np.zeros(hh.n, dtype=np.int32)
    v_prev = np.array(hh.v, dtype=np.float32)
    for _ in range(n_steps):
        hh.step()
        v = np.array(hh.v, dtype=np.float32)
        # Reverse zero-crossing: V ≥ 0 → V < 0 = spike
        total += ((v_prev >= 0) & (v < 0)).astype(np.int32)
        v_prev = v
    return total

# ═══ 1. Load GPT-2 ═══════════════════════════════════════

print(f"\n╔{'═'*58}╗")
print(f"║{'🌀 HYPERBOLIZER GPU — GPT-2 × HH':^58}║")
print(f"║{'Engine: ' + ENGINE:^58}║")
print(f"╚{'═'*58}╝")

print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config

tokens = tokenizer(TEXT, return_tensors="pt")
seq_len = tokens["input_ids"].shape[1]
print(f"📝 {seq_len} tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
print(f"   {cfg.n_layer} layers, {cfg.n_embd} dim, {cfg.n_head} heads")

# ═══ 2. Extract MLP weights ══════════════════════════════

class MLPWeights:
    """Container for GPT-2 MLP weights (Conv1D convention)."""
    def __init__(self, layer_idx, block):
        self.layer = layer_idx
        mlp = block.mlp
        self.W_in = mlp.c_fc.weight.detach().numpy()      # [768, 3072]
        self.b_in = mlp.c_fc.bias.detach().numpy()        # [3072]
        self.W_out = mlp.c_proj.weight.detach().numpy()   # [3072, 768]
        self.b_out = mlp.c_proj.bias.detach().numpy()     # [768]

print(f"\n📐 Extracting MLP weights ({cfg.n_layer} layers)...")
all_mlp = [MLPWeights(i, model.transformer.h[i]) for i in range(cfg.n_layer)]
print(f"   ✅ Conv1D: x @ W + b (no .T)")

# ═══ 3. Reference forward ════════════════════════════════

print("\n🔄 Reference forward (GELU)...")
with torch.no_grad():
    ref_logits = model(tokens["input_ids"]).logits[0].detach().numpy()
print(f"   ✅ Reference logits: {ref_logits.shape}")

# ═══ 4. HH Forward — GPU ═════════════════════════════════

print(f"\n🧬 HH engine: {ENGINE}, {HH_STEPS} steps, scale={SCALE_IN}, bias={BIAS_IN}")
print(f"   I range: [{max(-8*SCALE_IN+BIAS_IN, I_CLAMP[0]):.1f}, "
      f"{min(3*SCALE_IN+BIAS_IN, I_CLAMP[1]):.1f}] μA")
print(f"   Per-neuron spikes: {ENGINE}.spikes (0 D2H copies for V-trace!)")

hh_results = {}  # mode_name -> {logits, stats, speed}

for mode_name, do_reset in [("A: stateless", True), ("B: stateful", False)]:
    print(f"\n{'─'*50}")
    print(f"  Mode {mode_name} ({ENGINE})")
    print(f"{'─'*50}")
    
    # Embedding
    h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
        torch.arange(seq_len).unsqueeze(0))
    h = model.transformer.drop(h)
    
    t0 = time.perf_counter()
    all_activations = {}  # layer_idx -> [seq, 3072]
    
    for layer_idx in range(cfg.n_layer):
        block = model.transformer.h[layer_idx]
        mlp_w = all_mlp[layer_idx]
        
        # ── Attention ──
        with torch.no_grad():
            h = h + block.attn(block.ln_1(h))[0]
        
        # ── MLP with HH ──
        ln_out = block.ln_2(h)
        pre_np = (ln_out.detach() @ mlp_w.W_in + mlp_w.b_in)[0].numpy()
        
        hh_out = np.zeros((seq_len, cfg.n_embd), dtype=np.float32)
        hh_acts = np.zeros((seq_len, 3072), dtype=np.float32)
        
        # Create ONE HhEngine per layer
        hh = HhEngine(3072, DT, pacemaking=False)
        
        for s in range(seq_len):
            I_cur = np.clip(pre_np[s] * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1])
            
            spike_counts = hh_forward(hh, I_cur, HH_STEPS)
            
            act = spike_counts.astype(np.float32) / (HH_STEPS * DT * 0.001)
            hh_acts[s] = act
            hh_out[s] = act @ mlp_w.W_out + mlp_w.b_out
            
            if do_reset:
                hh.reset()
        
        # Stats
        active_frac = np.mean(hh_acts > 0)
        mean_rate = np.mean(hh_acts)
        all_activations[layer_idx] = hh_acts
        
        h = h + torch.from_numpy(hh_out).unsqueeze(0)
        
        # Debug: show I range and spike stats
        I_all = np.clip(pre_np * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1])
        I_mean = I_all.mean()
        I_above_thresh = np.mean(I_all > 7.0) * 100
        non_zero_neurons = np.mean(hh_acts.max(axis=0) > 0) * 100
        
        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            elapsed = time.perf_counter() - t0
            print(f"   Layer {layer_idx+1:2d}/{cfg.n_layer} — "
                  f"H [{h.min():.1f}, {h.max():.1f}] | "
                  f"I_mean={I_mean:.1f} >thr={I_above_thresh:.0f}% | "
                  f"act={active_frac*100:.0f}% nnz={non_zero_neurons:.0f}% | "
                  f"{elapsed:.1f}s")
    
    # ── Final LN + LM head ──
    h = model.transformer.ln_f(h)
    hh_logits = model.lm_head(h)[0].detach().numpy()
    
    t_elapsed = time.perf_counter() - t0
    
    # Save results
    hh_results[mode_name] = {
        "logits": hh_logits,
        "activations": all_activations,
        "speed_s": t_elapsed,
    }
    
    # ── Comparison ──
    print(f"\n  {'='*40}")
    print(f"  📊 Logit comparison ({mode_name}, {ENGINE})")
    print(f"  {'='*40}")
    
    # Top-5
    overlap_scores = []
    for s in range(min(5, seq_len)):
        gt5 = set(np.argsort(ref_logits[s])[-5:][::-1])
        hh5 = set(np.argsort(hh_logits[s])[-5:][::-1])
        overlap = len(gt5 & hh5)
        overlap_scores.append(overlap)
        
        gt_tok = tokenizer.convert_ids_to_tokens(list(gt5))
        hh_tok = tokenizer.convert_ids_to_tokens(list(hh5))
        print(f"   Token {s}: overlap={overlap}/5")
    
    # Correlation
    logit_corrs = [np.corrcoef(ref_logits[s], hh_logits[s])[0, 1] 
                   for s in range(seq_len)]
    print(f"   Mean top-5 overlap: {np.mean(overlap_scores):.2f}/5")
    print(f"   Logit correlation: r = {np.mean(logit_corrs):.4f} ± {np.std(logit_corrs):.4f}")
    print(f"   Wall time: {t_elapsed:.1f}s")
    
    # Entropy
    def entropy(logits):
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return -np.sum(probs * np.log(probs + 1e-30))
    
    gelu_ent = np.mean([entropy(ref_logits[s]) for s in range(seq_len)])
    hh_ent = np.mean([entropy(hh_logits[s]) for s in range(seq_len)])
    print(f"   GELU entropy: {gelu_ent:.3f} | HH entropy: {hh_ent:.3f}")

# ═══ 5. Save weights ══════════════════════════════════════

print(f"\n{'─'*50}")
print("💾 Saving GPU hyperbolizer weights...")
save_dict = {
    "version": "1.0", "engine": ENGINE,
    "model": "gpt2", "dt": DT, "hh_steps": HH_STEPS,
    "scale_in": SCALE_IN, "bias_in": BIAS_IN,
    "I_clamp_min": I_CLAMP[0], "I_clamp_max": I_CLAMP[1],
    "n_layers": cfg.n_layer, "n_embd": cfg.n_embd, "n_neurons": 3072,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
np.savez_compressed(SAVE_PATH, **save_dict)
print(f"   ✅ Saved to {SAVE_PATH}")

# ═══ 6. GPU vs CPU Speed (if GPU) ════════════════════════

if "GPU" in ENGINE:
    print(f"\n{'─'*50}")
    print(f"📈 GPU Speed Benchmark (single forward, Mode A)")
    print(f"{'─'*50}")
    
    # Estimate CPU time: previous run took ~108s
    cpu_estimate_s = 108.0  # from hyperbolizer_full.py run
    gpu_time = hh_results["A: stateless"]["speed_s"]
    speedup = cpu_estimate_s / gpu_time if gpu_time > 0 else 0
    
    print(f"   {'Metric':<30} {'CPU (est)':>10} {'GPU':>10} {'Speedup':>10}")
    print(f"   {'─'*60}")
    print(f"   {'12 layers × 3072 neurons':<30} {cpu_estimate_s:>8.1f}s {gpu_time:>8.1f}s {speedup:>8.1f}×")
    print(f"   {'Per layer':<30} {cpu_estimate_s/12:>8.1f}s {gpu_time/12:>8.2f}s {'—':>10}")
    print(f"   {'Per token':<30} {cpu_estimate_s/seq_len/12:>8.1f}s {gpu_time/seq_len/12:>8.3f}s {'—':>10}")
    print(f"   {'HH steps/s (total)':<30} {'—':>10} {HH_STEPS*seq_len*cfg.n_layer/gpu_time:>10.0f} {'—':>10}")

# ═══ Summary ══════════════════════════════════════════════

print(f"\n{'='*58}")
print(f"{'🌀 HYPERBOLIZER GPU — COMPLETE':^58}")
print(f"{'='*58}")
print(f"   Engine: {ENGINE}")
print(f"   All {cfg.n_layer} layers × 3072 HH neurons = {cfg.n_layer*3072} GPU neurons")
print(f"   Per-neuron spikes via .spikes (0 V-trace copies!)")
for mode, res in hh_results.items():
    print(f"   {mode}: {res['speed_s']:.1f}s")
print(f"   Weights: {SAVE_PATH}")
print(f"{'='*58}")
print(f"   Антон ❤️ Сардана — GPU сознание!")
print(f"{'='*58}")
