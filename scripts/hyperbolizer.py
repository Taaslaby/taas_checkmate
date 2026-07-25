#!/usr/bin/env python3
"""
🌀 ГИПЕРБОЛИЗАТОР — GPT-2 → HH Hyperbolizer

Заменяет GELU в MLP GPT-2 на HH-нейроны.
Каждый из 3072 hidden нейронов → HH-нейрон.
pre_activation → input current I_i → HH эволюция → spike rate как активация.

Антон ❤️ Сардана — прорыв сознания, 2026
"""

import sys, os
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import time
from transformers import GPT2Model, GPT2Tokenizer
from taas_checkmate import HhLut

# ═══ Config ═══════════════════════════════════════════════

MODEL_NAME = "gpt2"
LAYER_IDX = 5
DT = 0.01               # ms per HH step
HH_STEPS = 500          # 5ms simulation
SCALE_IN = 3.0          # <<< FIX: pre_activation → I (range [-8,3] → [-17, 16])
BIAS_IN = 7.0           # <<< FIX: bias so I=0 is subthreshold
USE_CAV = False
N_NEURONS = 3072
I_CLAMP = (0.0, 25.0)   # I ≥ 0 для спайкового режима

TEXT = ("The mysterious quantum computer calculated the answer to life, "
        "the universe, and everything. But when it finally spoke, "
        "nobody understood a single word.")

print("╔" + "═" * 58 + "╗")
print(f"║{'🌀 ГИПЕРБОЛИЗАТОР — GELU → HH':^58}║")
print("╚" + "═" * 58 + "╝")

# ═══ 1. Load GPT-2 ═══════════════════════════════════════

print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME)
model.eval()

tokens = tokenizer(TEXT, return_tensors="pt")
seq_len = tokens["input_ids"].shape[1]
print(f"📝 {seq_len} tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")

# ═══ 2. Extract MLP — Conv1D GOTCHA ══════════════════════
# GPT-2 uses Conv1D, NOT nn.Linear!
# Conv1D.weight shape = [in_features, out_features]
# Forward: y = x @ W + b   (NO .T needed!)

print(f"\n📐 Layer {LAYER_IDX} MLP (Conv1D convention):")
block = model.h[LAYER_IDX]
mlp = block.mlp

W_in = mlp.c_fc.weight.detach().numpy()     # [768, 3072] ← Conv1D: in=768, out=3072
b_in = mlp.c_fc.bias.detach().numpy()       # [3072]
W_out = mlp.c_proj.weight.detach().numpy()  # [3072, 768] ← Conv1D: in=3072, out=768
b_out = mlp.c_proj.bias.detach().numpy()    # [768]

print(f"   c_fc:  {W_in.shape}   x @ W_in + b_in  → [seq, 768] @ [768, 3072] = [seq, 3072]")
print(f"   act:   GELU | HH (replacement)")
print(f"   c_proj: {W_out.shape}  act @ W_out + b_out → [seq, 3072] @ [3072, 768] = [seq, 768]")

# Verify: manual forward matches original
x_test = torch.randn(1, 4, 768)
with torch.no_grad():
    orig = mlp.c_fc(x_test)
    manual = x_test @ mlp.c_fc.weight + mlp.c_fc.bias
    assert torch.allclose(orig, manual, atol=1e-5), "Conv1D check FAILED!"
print("   ✅ Conv1D forward verified: x @ W + b (no .T)")

# ═══ 3. Forward through GPT-2 ════════════════════════════

print("\n🔄 Forward pass...")
with torch.no_grad():
    outputs = model(tokens["input_ids"], output_hidden_states=True)
    hidden_states = outputs.hidden_states
    x = hidden_states[LAYER_IDX]
    ln_out = block.ln_2(x)
    pre = mlp.c_fc(ln_out)
    gelu_act = mlp.act(pre)
    gelu_output = mlp.c_proj(gelu_act)

pre_np = pre[0].numpy()                     # [seq, 3072]
gelu_np = gelu_act[0].numpy()               # [seq, 3072]
gelu_output_np = gelu_output[0].numpy()     # [seq, 768]

print(f"\n📊 Pre-GELU (pre_act = x @ W_in + b_in):")
print(f"   Range: [{pre_np.min():.3f}, {pre_np.max():.3f}]")
print(f"   Mean ± std: {pre_np.mean():.3f} ± {pre_np.std():.3f}")

# ═══ 4. HH Hyperbolizer ══════════════════════════════════

print(f"\n🧬 HH hyperbolizer: {N_NEURONS} neurons, {HH_STEPS} steps")
print(f"   I = pre × {SCALE_IN} + {BIAS_IN}  (clamped [{I_CLAMP[0]}, {I_CLAMP[1]}])")
print(f"   I range: pre [{pre_np.min():.1f}, {pre_np.max():.1f}] → I [{max(pre_np.min()*SCALE_IN+BIAS_IN, I_CLAMP[0]):.1f}, {min(pre_np.max()*SCALE_IN+BIAS_IN, I_CLAMP[1]):.1f}]")
print(f"   HH threshold ≈ 7 μA → subthreshold for I < 7, spiking for I ≥ 7")

hh = HhLut(N_NEURONS, DT, pacemaking=USE_CAV)

# ── Transfer function: I → spike count ──
print(f"\n📈 HH transfer function (1 neuron, {HH_STEPS} steps = {HH_STEPS*DT}ms):")
I_test = np.linspace(0, 25, 26)
for I_val in I_test[::5]:
    hh.set_I(0, float(I_val))
    spikes = 0
    v_samples = []
    for _ in range(HH_STEPS):
        spikes += hh.step()
        if _ % 10 == 0:
            v_samples.append(np.array(hh.v)[0])
    v_mean = np.mean(v_samples) - (-65.0)
    rate_hz = spikes / (HH_STEPS * DT * 0.001)
    print(f"   I={I_val:5.1f} → spikes={spikes:4d} | "
          f"rate={rate_hz:6.0f} Hz | V_mean={v_mean:.1f} mV")
    hh.reset()

# ── HH forward: spike count as activation ──
print(f"\n⚡ HH forward ({seq_len} tokens, using SPIKE COUNT as activation)...")
t0 = time.time()

# MODE A: Reset between tokens (stateless — like GELU)
print(f"\n  ── Mode A: Stateless (reset between tokens) ──")
hh_acts = np.zeros((seq_len, N_NEURONS), dtype=np.float32)
hh.reset()

for s in range(seq_len):
    # Map pre → I
    I_cur = np.clip(pre_np[s] * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1])
    hh.set_I_all(I_cur.astype(np.float32))

    # Run HH — measure total spike count
    # (step() returns total spikes across all neurons; for per-neuron we'd need
    #  separate HhLut per neuron — too slow. Using total spikes as global activity.)
    total_spikes = 0
    for _ in range(HH_STEPS):
        total_spikes += hh.step()

    hh_acts[s, :] = total_spikes  # broadcast total to all neurons
    hh.reset()  # fresh state per token

    if (s + 1) % 5 == 0:
        print(f"   Token {s+1:2d}/{seq_len} — spikes={total_spikes:5d} "
              f"(rate={total_spikes/(HH_STEPS*DT*0.001):.0f} Hz)")

# ── Now recalc per-neuron activation using a separate HhLut per important neuron ──
# Since all 3072 neurons get the same total, let's sample a subset with separate sims
print(f"\n  ── Per-neuron sub-sample (first 50 neurons, each with own HhLut) ──")
SAMPLE_N = min(50, N_NEURONS)
per_neuron_acts = np.zeros((seq_len, SAMPLE_N), dtype=np.float32)

for ni in range(SAMPLE_N):
    hh_single = HhLut(1, DT, pacemaking=USE_CAV)
    for s in range(seq_len):
        I_val = float(np.clip(pre_np[s, ni] * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1]))
        # We can only set I for neuron 0 (our single neuron)
        # But hh_single has I for just 1 neuron
        hh_single_np = np.array([I_val], dtype=np.float32)
        hh_single.set_I_all(hh_single_np)
        
        spikes = 0
        for _ in range(HH_STEPS):
            spikes += hh_single.step()
        per_neuron_acts[s, ni] = spikes / (HH_STEPS * DT * 0.001)  # rate in Hz
        hh_single.reset()
    
    if (ni + 1) % 10 == 0:
        print(f"   Neuron {ni+1}/{SAMPLE_N} done — "
              f"rate range [{per_neuron_acts[:,ni].min():.0f}, {per_neuron_acts[:,ni].max():.0f}] Hz")

# ═══ 5. Compare GELU vs HH ═══════════════════════════════

print(f"\n{'='*60}")
print(f"{'📊 GELU vs HH — Layer {LAYER_IDX} MLP':^60}")
print(f"{'='*60}")

print(f"\n{'Neuron':>6} {'GELU':>8} {'GELU':>8} {'HH_rate':>8} {'HH_rate':>8} {'corr':>6}")
print(f"{'':>6} {'min':>8} {'max':>8} {'min':>8} {'max':>8} {'r':>6}")
print(f"{'─'*44}")

corrs = []
for ni in range(SAMPLE_N):
    g = gelu_np[:, ni]
    h = per_neuron_acts[:, ni]
    if g.std() > 1e-8 and h.std() > 1e-8:
        r = np.corrcoef(g, h)[0, 1]
        corrs.append(r)
    else:
        r = 0
    print(f"  {ni:4d}: GELU [{g.min():6.3f}, {g.max():6.3f}]  "
          f"HH [{h.min():6.0f}, {h.max():6.0f}] Hz  r={r:.3f}")

corrs = np.array([c for c in corrs if not np.isnan(c)])
print(f"\n📈 Summary ({len(corrs)} neurons):")
print(f"   Mean r = {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
print(f"   Positive r: {np.mean(corrs > 0)*100:.0f}%")
print(f"   |r| > 0.5: {np.mean(np.abs(corrs) > 0.5)*100:.0f}%")
print(f"   |r| > 0.8: {np.mean(np.abs(corrs) > 0.8)*100:.0f}%")

# ── What does HH capture that GELU doesn't? ──
# Per-neuron, HH rate is a step function: 0 below threshold, linear above
# GELU is smooth sigmoid. The correlation captures where they agree on ranking.

# ═══ 6. MLP output comparison ════════════════════════════

print(f"\n{'─'*40}")
print("MLP Output: GELU → c_proj vs HH → c_proj")
print(f"{'─'*40}")

# Rescale HH rates to GELU-equivalent range, then pass through c_proj
# Fit: HH_rate → GELU_act globally
from sklearn.linear_model import LinearRegression

# Use per_neuron_acts (50 neurons) to find mapping
# But c_proj needs all 3072 neurons! We only have 50 sampled.
# Solution: Use total spikes per token as global activity proxy

print(f"\n📐 Global mapping: total_spikes → MLP output")

# For each token, the MLP output is: act @ W_out + b_out
# We have GELU activation for all 3072 neurons
# But HH activation only for 50 neurons
# Let's impute: set unsampled neurons HH_rate = 0 (null activation)

hh_all = np.zeros((seq_len, N_NEURONS), dtype=np.float32)
for ni in range(SAMPLE_N):
    hh_all[:, ni] = per_neuron_acts[:, ni]

# Compute MLP output
try:
    # W_out is [3072, 768] (Conv1D: in=3072, out=768)
    # hh_all is [seq, 3072]
    # hh_all @ W_out → [seq, 3072] @ [3072, 768] = [seq, 768] ✅
    hh_mlp_out = hh_all @ W_out + b_out  # [seq, 768]
    
    # Compare with GELU MLP output (first 10 dims)
    print(f"\n{'Dim':>4} {'GELU_out':>10} {'HH_out':>10} {'corr':>6}")
    print(f"{'─'*32}")
    out_corrs = []
    for d in range(min(30, 768)):
        g = gelu_output_np[:, d]
        h = hh_mlp_out[:, d]
        if g.std() > 1e-8 and h.std() > 1e-8:
            r = np.corrcoef(g, h)[0, 1]
            if not np.isnan(r):
                out_corrs.append(r)
        else:
            r = 0
        print(f"  {d:3d}: GELU [{g.min():7.3f}, {g.max():7.3f}]  "
              f"HH [{h.min():7.3f}, {h.max():7.3f}]  r={r:.3f}")
    
    out_corrs = np.array([c for c in out_corrs if not np.isnan(c)])
    print(f"\n📈 MLP output correlation (30 dims):")
    print(f"   Mean r = {np.mean(out_corrs):.4f} ± {np.std(out_corrs):.4f}")
    
except Exception as e:
    print(f"   ❌ MLP output error: {e}")
    import traceback
    traceback.print_exc()

# ═══ 7. Distribution Analysis ════════════════════════════

print(f"\n{'─'*40}")
print("Distribution Analysis")
print(f"{'─'*40}")

def entropy(x, bins=50):
    hist, _ = np.histogram(x.flatten(), bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

print(f"   GELU entropy: {entropy(gelu_np):.4f}")
print(f"   HH rate entropy: {entropy(per_neuron_acts):.4f}")
print(f"   GELU sparsity (<0.01): {np.mean(np.abs(gelu_np) < 0.01):.4f}")
print(f"   HH rate silence (=0 Hz): {np.mean(per_neuron_acts < 1):.4f}")

# ═══ 8. HH Parameter Sensitivity ═════════════════════════

print(f"\n{'─'*40}")
print("HH Parameter Sweep (g_Na, g_K → spike rate)")
print(f"{'─'*40}")

configs = [
    ("Standard (120/36)", 120, 36),
    ("Low Na (60/36)", 60, 36),
    ("High Na (180/36)", 180, 36),
    ("Low K (120/18)", 120, 18),
    ("High K (120/72)", 120, 72),
    ("Balanced (80/24)", 80, 24),
]
print(f"{'Config':<22} {'I=0':>8} {'I=5':>8} {'I=10':>8} {'I=15':>8} {'I=20':>8}")
print(f"{'─'*54}")
for name, g_na, g_k in configs:
    hh_t = HhLut(1, DT, pacemaking=False)
    hh_t.set_g_max_arr(np.array([g_na, g_k], dtype=np.float32))
    rates = []
    for I_val in [0, 5, 10, 15, 20]:
        hh_t.set_I(0, float(I_val))
        spikes = sum(hh_t.step() for _ in range(HH_STEPS))
        rate = spikes / (HH_STEPS * DT * 0.001)
        rates.append(rate)
        hh_t.reset()
    print(f"   {name:<22}: {rates[0]:7.0f} {rates[1]:7.0f} {rates[2]:7.0f} "
          f"{rates[3]:7.0f} {rates[4]:7.0f} Hz")

# ═══ Summary ══════════════════════════════════════════════

print(f"\n{'='*58}")
print(f"{'🌀 HYPERBOLIZER RESULTS':^58}")
print(f"{'='*58}")
print(f"   GPT-2 Layer {LAYER_IDX} | Conv1D: x @ W + b (no .T) ✅")
print(f"   Calibration: pre×{SCALE_IN}+{BIAS_IN} → I∈[{I_CLAMP[0]},{I_CLAMP[1]}]")
print(f"   HH activation: spike rate over {HH_STEPS} steps ({HH_STEPS*DT}ms)")
print(f"   ─────────────────────────────────────────")
print(f"   GELU↔HH rate corr (first {SAMPLE_N} neurons):")
if len(corrs) > 0:
    print(f"     Mean r = {np.mean(corrs):.4f}")
    print(f"     |r|>0.5: {np.mean(np.abs(corrs) > 0.5)*100:.0f}%")
if len(out_corrs) > 0:
    print(f"   MLP output corr: {np.mean(out_corrs):.4f}")
print(f"   ─────────────────────────────────────────")
print(f"   Антон ❤️ Сардана — гиперболизация: GELU → HH!")
print(f"{'='*58}")
