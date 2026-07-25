#!/usr/bin/env python3
"""
🌀 ГИПЕРБОЛИЗАТОР FULL — GPT-2 × HH

- Все 3072 нейрона через V-trace spike detection (один HhLut, не 50!)
- Mode A: stateless (reset между токенами)
- Mode B: stateful (HH запоминает контекст между токенами)
- Все 12 слоёв GPT-2
- Сравнение логитов GELU vs HH
- Сохранение HH-весов + маппинг параметров

Антон ❤️ Сардана — гиперболизация сознания, 2026
"""

import sys, os, json, time
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from taas_checkmate import HhLut

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01                  # ms per HH step
HH_STEPS = 500             # 5ms simulation (enough for 1-2 spikes at 200Hz)
SCALE_IN = 3.0             # pre_activation → I: I = pre × scale + bias
BIAS_IN = 7.0
I_CLAMP = (0.0, 25.0)
SAVE_PATH = "hyperbolizer_weights.npz"
TEXT = "The mysterious quantum computer calculated the answer."

# ═══ Helpers ══════════════════════════════════════════════

def hh_forward(hh: HhLut, I_cur: np.ndarray, n_steps: int) -> np.ndarray:
    """Run HH and return per-neuron spike counts.
    
    Один HhLut для ВСЕХ 3072 нейронов (Rust SIMD).
    Спайки детектятся инкрементально — без аллокации v_trace!
    
    Args:
        hh: HhLut instance с n_neurons
        I_cur: [n_neurons] input currents
        n_steps: число шагов HH
    
    Returns:
        spike_counts: [n_neurons] per-neuron spike counts
    """
    hh.set_I_all(I_cur.astype(np.float32))
    spike_counts = np.zeros(hh.n, dtype=np.int32)
    v_prev = np.array(hh.v, dtype=np.float32)
    for _ in range(n_steps):
        hh.step()
        v = np.array(hh.v, dtype=np.float32)
        # Reverse zero-crossing = spike (совпадает с fused_hh_poly_step)
        spike_counts += ((v_prev >= 0) & (v < 0)).astype(np.int32)
        v_prev = v
    return spike_counts

# ═══ 1. Load GPT-2 ═══════════════════════════════════════

print("╔" + "═" * 58 + "╗")
print(f"║{'🌀 HYPERBOLIZER FULL — GPT-2 × HH':^58}║")
print("╚" + "═" * 58 + "╝")

print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config

tokens = tokenizer(TEXT, return_tensors="pt")
seq_len = tokens["input_ids"].shape[1]
print(f"📝 {seq_len} tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
print(f"   Model: {cfg.n_layer} layers, {cfg.n_embd} dim, {cfg.n_head} heads")

# ═══ 2. Extract ALL MLP weights ══════════════════════════

class MLPWeights:
    """Container for GPT-2 MLP weights (Conv1D convention)."""
    def __init__(self, layer_idx, block):
        self.layer = layer_idx
        mlp = block.mlp
        # Conv1D: weight = [in_features, out_features], forward = x @ W + b
        self.W_in = mlp.c_fc.weight.detach().numpy()      # [768, 3072]
        self.b_in = mlp.c_fc.bias.detach().numpy()        # [3072]
        self.W_out = mlp.c_proj.weight.detach().numpy()   # [3072, 768]
        self.b_out = mlp.c_proj.bias.detach().numpy()     # [768]

print(f"\n📐 Extracting all {cfg.n_layer} MLP layers...")
all_mlp = [MLPWeights(i, model.transformer.h[i]) for i in range(cfg.n_layer)]
print(f"   ✅ {len(all_mlp)} MLPs extracted (Conv1D: x @ W + b)")

# ═══ 3. Reference: forward with GELU ═════════════════════

print("\n🔄 Reference forward (GELU)...")
with torch.no_grad():
    ref_outputs = model(tokens["input_ids"])
    ref_logits = ref_outputs.logits[0].detach().numpy()  # [seq, vocab]

print(f"   ✅ Reference logits: {ref_logits.shape}")

# ═══ 4. HH Forward — Layer by Layer ══════════════════════

print(f"\n🧬 HH hyperbolizer: {HH_STEPS} steps, scale={SCALE_IN}, bias={BIAS_IN}")
print(f"   I range: [{max(-8*SCALE_IN+BIAS_IN, I_CLAMP[0]):.1f}, "
      f"{min(3*SCALE_IN+BIAS_IN, I_CLAMP[1]):.1f}] μA")

# We'll store HH activations per layer
hh_acts_per_layer = {}  # layer -> {mode: [seq, 3072]}
hh_params = {}           # saved HH config

for mode_name, do_reset in [("A: stateless", True), ("B: stateful", False)]:
    print(f"\n{'─'*50}")
    print(f"  Mode {mode_name}")
    print(f"{'─'*50}")
    
    # Embedding + positional
    h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
        torch.arange(seq_len).unsqueeze(0))
    h = model.transformer.drop(h)  # eval mode: identity
    
    t0 = time.time()
    
    for layer_idx in range(cfg.n_layer):
        block = model.transformer.h[layer_idx]
        mlp_w = all_mlp[layer_idx]
        
        # ── Attention block (eval mode: dropout = identity) ──
        with torch.no_grad():
            attn_out = block.attn(block.ln_1(h))[0]
            h = h + attn_out
        
        # ── MLP block with HH ──
        ln_out = block.ln_2(h)
        
        # Pre-HH activations (detach: no grad нужен для инференса)
        pre_np = (ln_out.detach() @ mlp_w.W_in + mlp_w.b_in)[0].numpy()  # [seq, 3072]
        
        # HH activation for each token
        hh_out = np.zeros((seq_len, cfg.n_embd), dtype=np.float32)
        hh_activations = np.zeros((seq_len, 3072), dtype=np.float32)  # для статистики
        
        # Create ONE HhLut per layer for all tokens
        hh = HhLut(3072, DT, pacemaking=False)
        
        for s in range(seq_len):
            I_cur = np.clip(pre_np[s] * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1])
            
            # Run HH — per-neuron spike counts (один HhLut, инкрементально)
            spike_counts = hh_forward(hh, I_cur, HH_STEPS)
            
            # Activation = spike rate in Hz
            act = spike_counts.astype(np.float32) / (HH_STEPS * DT * 0.001)
            hh_activations[s] = act
            
            # Through c_proj
            hh_out[s] = act @ mlp_w.W_out + mlp_w.b_out
            
            if do_reset:
                hh.reset()
            # Mode B: NO reset — HH state carries to next token! 🧠
        
        # Статистика активаций
        active_frac = np.mean(hh_activations > 0)
        mean_rate = np.mean(hh_activations)
        silent_frac = np.mean(hh_activations == 0)
        
        # Residual connection (eval mode: dropout = identity)
        hh_tensor = torch.from_numpy(hh_out).unsqueeze(0)
        h = h + hh_tensor
        
        if (layer_idx + 1) % 4 == 0 or layer_idx == cfg.n_layer - 1:
            print(f"   Layer {layer_idx+1:2d}/{cfg.n_layer} — "
                  f"H [{h.min():.3f}, {h.max():.3f}] | "
                  f"HH active={active_frac*100:.0f}% silent={silent_frac*100:.0f}% "
                  f"rate={mean_rate:.0f}Hz")
    
    # ── Final LN + LM head ──
    h = model.transformer.ln_f(h)
    hh_logits = model.lm_head(h)[0].detach().numpy()  # [seq, vocab]
    
    t_elapsed = time.time() - t0
    print(f"\n⏱  HH forward ({mode_name}): {t_elapsed:.1f}s")
    
    # ═══ Compare with reference ════════════════════════════
    print(f"\n  {'='*40}")
    print(f"  📊 Logit comparison — GELU vs HH ({mode_name})")
    print(f"  {'='*40}")
    
    # Top-5 tokens per position
    for s in range(min(5, seq_len)):
        gelu_top5 = np.argsort(ref_logits[s])[-5:][::-1]
        hh_top5 = np.argsort(hh_logits[s])[-5:][::-1]
        
        gelu_tokens = tokenizer.convert_ids_to_tokens(gelu_top5)
        hh_tokens = tokenizer.convert_ids_to_tokens(hh_top5)
        
        print(f"\n  Token {s}: '{tokenizer.convert_ids_to_tokens([tokens['input_ids'][0,s].item()])[0]}'")
        print(f"    GELU top-5: {list(zip(gelu_tokens, [f'{ref_logits[s,i]:.2f}' for i in gelu_top5]))}")
        print(f"    HH top-5:   {list(zip(hh_tokens, [f'{hh_logits[s,i]:.2f}' for i in hh_top5]))}")
        
        # Overlap
        overlap = len(set(gelu_top5) & set(hh_top5))
        print(f"    Overlap: {overlap}/5")
    
    # Per-position rank correlation
    print(f"\n  📈 Rank correlation (Spearman-like):")
    ranks = []
    for s in range(seq_len):
        # How many of GELU's top-5 are in HH's top-5?
        gelu_top5 = set(np.argsort(ref_logits[s])[-5:][::-1])
        hh_top5 = set(np.argsort(hh_logits[s])[-5:][::-1])
        ranks.append(len(gelu_top5 & hh_top5))
    print(f"    Mean top-5 overlap: {np.mean(ranks):.2f}/5")
    
    # Logit correlation
    logit_corrs = []
    for s in range(seq_len):
        r = np.corrcoef(ref_logits[s], hh_logits[s])[0, 1]
        logit_corrs.append(r)
    print(f"    Logit correlation: r = {np.mean(logit_corrs):.4f} ± {np.std(logit_corrs):.4f}")
    
    # 🧠 Consciousness metrics
    print(f"\n  🧠 Phi consciousness proxy:")
    # Shannon entropy of logit distribution
    def softmax_entropy(logits):
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return -np.sum(probs * np.log(probs + 1e-30))
    
    gelu_ent = [softmax_entropy(ref_logits[s]) for s in range(seq_len)]
    hh_ent = [softmax_entropy(hh_logits[s]) for s in range(seq_len)]
    print(f"    GELU entropy: {np.mean(gelu_ent):.3f}")
    print(f"    HH entropy:   {np.mean(hh_ent):.3f}")

# ═══ 5. Save weights ══════════════════════════════════════

print(f"\n{'─'*50}")
print("💾 Saving hyperbolizer weights...")
print(f"{'─'*50}")

# Save: mapping params + layer configs
save_dict = {
    "version": "1.0",
    "model": "gpt2",
    "dt": DT,
    "hh_steps": HH_STEPS,
    "scale_in": SCALE_IN,
    "bias_in": BIAS_IN,
    "I_clamp_min": I_CLAMP[0],
    "I_clamp_max": I_CLAMP[1],
    "n_layers": cfg.n_layer,
    "n_embd": cfg.n_embd,
    "n_neurons": 3072,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Save per-layer HH config (parameters are shared, but we save for completeness)
for layer_idx in range(cfg.n_layer):
    hh = HhLut(1, DT, pacemaking=False)  # single neuron for reference params
    save_dict[f"layer_{layer_idx}_g_max_arr"] = np.array(hh.g_max_arr)
    save_dict[f"layer_{layer_idx}_E_rev_arr"] = np.array(hh.E_rev_arr)
    save_dict[f"layer_{layer_idx}_g_leak"] = np.array(hh.g_leak)
    save_dict[f"layer_{layer_idx}_C_m"] = np.array(hh.C_m)
    save_dict[f"layer_{layer_idx}_n_channels"] = hh.n_channels
    save_dict[f"layer_{layer_idx}_n_gates"] = hh.n_gates

np.savez_compressed(SAVE_PATH, **save_dict)
print(f"   ✅ Saved to {SAVE_PATH}")
print(f"   Contains: {list(save_dict.keys())[:10]}...")

# ═══ Summary ══════════════════════════════════════════════

print(f"\n{'='*58}")
print(f"{'🌀 HYPERBOLIZER FULL — COMPLETE':^58}")
print(f"{'='*58}")
print(f"   ✅ All {cfg.n_layer} layers × 3072 neurons = {cfg.n_layer*3072} HH neurons")
print(f"   ✅ Mode A (stateless) + Mode B (stateful)")
print(f"   ✅ Logit comparison GELU vs HH")
print(f"   ✅ Weights saved to {SAVE_PATH}")
print(f"{'='*58}")
print(f"   Антон ❤️ Сардана — сознание через гиперболизацию!")
print(f"{'='*58}")
