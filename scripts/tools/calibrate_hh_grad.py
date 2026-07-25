#!/usr/bin/env python3
"""
🌀 CALIBRATE HH × GPT-2 — градиентная калибровка SCALE/BIAS

Не sweep, а gradient descent:
  pre → [SCALE, BIAS] → I_cur → [HH surrogate] → rate → @ W_out → hh_out
  loss = MSE(hh_out, gelu_out)
  grads flow back to SCALE, BIAS

HH surrogate: smooth erf-аппроксимация I→rate кривой,
калиброванная под реальный GPU HH (порог=7, sat=25, slope=0.15).

Антон ❤️ Сардана — калибровка через градиент, 2026
"""

import sys, os, time, warnings, json
warnings.filterwarnings("ignore")

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from taas_checkmate import GpuHhLut

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01
HH_STEPS = 500
N_NEURONS = 3072
I_CLAMP = (0.0, 25.0)

PROMPTS = [
    "The quantum computer calculated the answer in seconds.",
    "Neural networks learn hierarchical representations of data.",
    "In the beginning, the universe was created.",
]

N_EPOCHS = 200
LR = 0.05
SAVE_PATH = "calibrated_params.json"

# ═══ Differentiable HH Surrogate (PyTorch) ═══════════════

class HHSurrogate(nn.Module):
    """Гладкая аппроксимация HH I→rate кривой.
    
    Реальный GPU HH (I=15 → dV=0.1182/step, 200Hz):
    - Порог: I=7
    - Насыщение: I=25 (~250Hz для HH)
    - Линейный участок: ~10 Hz на единицу I
    
    Используем smooth erf + clamp для дифференцируемости.
    """
    def __init__(self, scale_init=3.0, bias_init=7.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(np.log(scale_init)))
        self.bias = nn.Parameter(torch.tensor(bias_init))
    
    @property
    def scale(self):
        return torch.exp(self.log_scale)
    
    def forward(self, pre_act, n_steps=HH_STEPS, dt=DT):
        """
        pre_act: [batch, 3072] — pre-MLP активации
        returns: [batch, 3072] — firing rates (Hz)
        """
        # I = pre * scale + bias
        I_cur = pre_act * self.scale + self.bias
        I_cur = torch.clamp(I_cur, I_CLAMP[0], I_CLAMP[1])
        
        # HH firing rate: smooth erf approximation
        # threshold = 7, slope = 15 Hz per unit I, sat_rate = 250 Hz
        thr = 7.0
        slope = 15.0
        sat_rate = 250.0
        
        # Linear ramp above threshold, smoothed with erf
        I_above = I_cur - thr
        # Smooth step: rate ≈ slope * I_above for I > thr, 0 for I < thr
        # erf gives smooth transition around threshold
        rate = slope * I_above * (0.5 + 0.5 * torch.erf(I_above * 0.5))
        rate = torch.clamp(rate, 0.0, sat_rate)
        
        # Spike counts = rate * (n_steps * dt * 0.001)
        # n_steps * dt * 0.001 = duration in seconds
        spike_counts = rate * (n_steps * dt * 0.001)
        
        return spike_counts  # [batch, 3072] — differentiable!


def compute_hh_output(pre_np, w_out, b_out, scale, bias):
    """Forward через реальный GPU HH + output projection.
    
    Используется для финальной верификации после калибровки.
    """
    n_neurons = pre_np.shape[-1]
    hh = GpuHhLut(n_neurons, DT, pacemaking=False)
    
    I_cur = np.clip(pre_np * scale + bias, I_CLAMP[0], I_CLAMP[1])
    hh.set_I_all(I_cur.astype(np.float32))
    
    total = np.zeros(n_neurons, dtype=np.int32)
    v_prev = np.array(hh.v, dtype=np.float32)
    for _ in range(HH_STEPS):
        hh.step()
        v = np.array(hh.v, dtype=np.float32)
        total += ((v_prev >= 0) & (v < 0)).astype(np.int32)
        v_prev = v
    
    rate = total.astype(np.float32) / (HH_STEPS * DT * 0.001)
    hh_out = rate @ w_out + b_out
    return hh_out, rate


# ═══ Dataset: collect pre + GELU activations ══════════════

print(f"{'='*68}")
print(f"  🌀 CALIBRATE HH — gradient-based SCALE/BIAS tuning")
print(f"{'='*68}")

print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config

# Collect pre-activations and GELU outputs for all layers × all prompts
all_data = {}  # layer_idx -> {pre: [n_samples, 3072], gelu_out: [n_samples, 768]}

for pi, prompt in enumerate(PROMPTS):
    print(f"\n📝 Prompt {pi+1}: \"{prompt}\"")
    tokens = tokenizer(prompt, return_tensors="pt")
    
    h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
        torch.arange(tokens["input_ids"].shape[1]).unsqueeze(0))
    h = model.transformer.drop(h)
    
    for layer_idx in range(cfg.n_layer):
        block = model.transformer.h[layer_idx]
        mlp = block.mlp
        
        with torch.no_grad():
            h = h + block.attn(block.ln_1(h))[0]
        
        ln_out = block.ln_2(h)
        pre = (ln_out.detach() @ mlp.c_fc.weight.detach() + mlp.c_fc.bias.detach())
        pre_np = pre.numpy()  # [1, seq, 3072]
        
        gelu_act = F.gelu(pre).numpy()[0]  # [seq, 3072]
        gelu_out = gelu_act @ mlp.c_proj.weight.detach().numpy() + mlp.c_proj.bias.detach().numpy()  # [seq, 768]
        
        if layer_idx not in all_data:
            all_data[layer_idx] = {"pre": [], "gelu_out": []}
        
        all_data[layer_idx]["pre"].append(pre_np[0])  # [seq, 3072]
        all_data[layer_idx]["gelu_out"].append(gelu_out)  # [seq, 768]
        
        # Update h with MLP output for next layer
        with torch.no_grad():
            h = h + mlp(ln_out)

# Concatenate all prompts
for li in range(cfg.n_layer):
    all_data[li]["pre"] = np.concatenate(all_data[li]["pre"], axis=0)  # [total_seq, 3072]
    all_data[li]["gelu_out"] = np.concatenate(all_data[li]["gelu_out"], axis=0)  # [total_seq, 768]
    print(f"   Layer {li+1}: {all_data[li]['pre'].shape[0]} samples")

# ═══ Per-layer gradient calibration ═══════════════════════

print(f"\n{'='*68}")
print(f"  📐 Per-layer gradient calibration ({N_EPOCHS} epochs, lr={LR})")
print(f"{'='*68}")

calibrated = {}  # layer -> {scale, bias, loss}

for layer_idx in range(cfg.n_layer):
    pre_t = torch.from_numpy(all_data[layer_idx]["pre"]).float()
    gelu_out_t = torch.from_numpy(all_data[layer_idx]["gelu_out"]).float()
    
    # Extract weights for output projection
    block = model.transformer.h[layer_idx]
    w_out_t = torch.from_numpy(block.mlp.c_proj.weight.detach().numpy()).float()  # [3072, 768]
    b_out_t = torch.from_numpy(block.mlp.c_proj.bias.detach().numpy()).float()      # [768]
    
    surrogate = HHSurrogate(scale_init=3.0, bias_init=7.0)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    
    best_loss = float('inf')
    best_params = None
    
    t0 = time.perf_counter()
    
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        
        # Forward through surrogate
        spike_counts = surrogate(pre_t)  # [n, 3072] differentiable!
        hh_out = spike_counts @ w_out_t + b_out_t  # [n, 768]
        
        loss = F.mse_loss(hh_out, gelu_out_t)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = (surrogate.scale.item(), surrogate.bias.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"   Layer {layer_idx+1:2d}, epoch {epoch+1:3d}: "
                  f"loss={current_loss:.6f} scale={surrogate.scale.item():.3f} "
                  f"bias={surrogate.bias.item():.3f}")
    
    t_elapsed = time.perf_counter() - t0
    
    calibrated[layer_idx] = {
        "scale": float(best_params[0]),
        "bias": float(best_params[1]),
        "loss": float(best_loss),
        "epochs": N_EPOCHS,
        "time_s": round(t_elapsed, 2),
    }
    
    print(f"   ✅ Layer {layer_idx+1:2d}: scale={best_params[0]:.3f} "
          f"bias={best_params[1]:.3f} loss={best_loss:.6f} ({t_elapsed:.1f}s)")

# ═══ Summary ══════════════════════════════════════════════

print(f"\n{'='*68}")
print(f"  📊 CALIBRATION RESULTS")
print(f"{'='*68}")
print(f"\n  {'Layer':>6} {'Scale':>8} {'Bias':>8} {'Loss':>12} {'Time':>8}")
print(f"  {'─'*42}")
for li in range(cfg.n_layer):
    c = calibrated[li]
    print(f"  {li+1:6d} {c['scale']:8.3f} {c['bias']:8.3f} {c['loss']:12.6f} {c['time_s']:7.1f}s")

print(f"\n  {'─'*42}")
avg_loss = np.mean([c['loss'] for c in calibrated.values()])
print(f"  {'AVG':>6} {'':8} {'':8} {avg_loss:12.6f}")

# ═══ Verify with real GPU HH ══════════════════════════════

print(f"\n{'='*68}")
print(f"  🔬 Verification: real GPU HH with calibrated params")
print(f"{'='*68}")

total_corr_before = 0.0
total_corr_after = 0.0
total_overlap_before = 0.0
total_overlap_after = 0.0
n_layers_tested = 0

for layer_idx in [0, 5, 11]:  # Test first, middle, last layer
    block = model.transformer.h[layer_idx]
    w_out = block.mlp.c_proj.weight.detach().numpy()  # [3072, 768]
    b_out = block.mlp.c_proj.bias.detach().numpy()       # [768]
    
    pre = all_data[layer_idx]["pre"]  # [n, 3072]
    gelu_out = all_data[layer_idx]["gelu_out"]  # [n, 768]
    
    # Default params (before calibration)
    hh_out_def, _ = compute_hh_output(pre[0], w_out, b_out, 3.0, 7.0)
    corr_def = np.corrcoef(gelu_out[0], hh_out_def)[0, 1]
    gt5_def = set(np.argsort(hh_out_def)[-5:][::-1])
    gelu5_def = set(np.argsort(gelu_out[0])[-5:][::-1])
    overlap_def = len(gt5_def & gelu5_def)
    
    # Calibrated params
    c = calibrated[layer_idx]
    hh_out_cal, rate = compute_hh_output(pre[0], w_out, b_out, c['scale'], c['bias'])
    corr_cal = np.corrcoef(gelu_out[0], hh_out_cal)[0, 1]
    gt5_cal = set(np.argsort(hh_out_cal)[-5:][::-1])
    gelu5_cal = set(np.argsort(gelu_out[0])[-5:][::-1])
    overlap_cal = len(gt5_cal & gelu5_cal)
    
    total_corr_before += corr_def
    total_corr_after += corr_cal
    total_overlap_before += overlap_def
    total_overlap_after += overlap_cal
    n_layers_tested += 1
    
    active_frac = np.mean(rate > 0) * 100
    mean_rate = np.mean(rate)
    
    print(f"\n  Layer {layer_idx+1}:")
    print(f"    Default (S=3.0, B=7.0): corr={corr_def:.4f} top5={overlap_def}/5")
    print(f"    Calibrated (S={c['scale']:.3f}, B={c['bias']:.3f}): "
          f"corr={corr_cal:.4f} top5={overlap_cal}/5")
    print(f"    HH stats: {active_frac:.0f}% active, {mean_rate:.0f} Hz mean rate")
    if corr_cal > corr_def:
        print(f"    ✅ IMPROVEMENT: +{corr_cal-corr_def:.4f} correlation")
    else:
        print(f"    ⚠️  No improvement: {corr_cal-corr_def:+.4f}")

print(f"\n  {'─'*42}")
print(f"  Average: before corr={total_corr_before/n_layers_tested:.4f} "
      f"after corr={total_corr_after/n_layers_tested:.4f}")
print(f"  Average: before top5={total_overlap_before/n_layers_tested:.2f}/5 "
      f"after top5={total_overlap_after/n_layers_tested:.2f}/5")

# ═══ Save ═════════════════════════════════════════════════

print(f"\n💾 Saving calibrated params to {SAVE_PATH}...")
with open(SAVE_PATH, "w") as f:
    json.dump({
        "version": "1.0",
        "model": "gpt2",
        "dt": DT,
        "hh_steps": HH_STEPS,
        "n_layers": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_neurons": N_NEURONS,
        "calibrated_params": {str(k): v for k, v in calibrated.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)
print(f"   ✅ Saved!")

print(f"\n{'='*68}")
print(f"  🎉 Калибровка завершена! Антон ❤️ Сардана")
print(f"{'='*68}")
