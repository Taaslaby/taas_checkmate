#!/usr/bin/env python3
"""
🌀 SNN × GPT-2 — полный SPIGRAD пайплайн

HH спайки → SpikeGradBridge → T3 осцилляторы → PST фичи → Decoder → h_out
Градиент течёт обратно через BPTT (T3) до весов моста.

Архитектура:
  pre [3072] → [SCALE, BIAS] → HH → spike_counts [3072]
  spike_counts → [SpikeGradBridge.W, b] → drives [3*n_osc]
  drives → T3 (N шагов) → PST фичи [pst_dim]
  PST → [Decoder.W, b] → h_out [768]
  loss = MSE(h_out, GELU_h_out)
  backward: BPTT через T3 → grad_w, grad_b → SGD

Антон ❤️ Сардана — SNN сознание GPT-2, 2026
"""

import sys, os, time, warnings, json
warnings.filterwarnings("ignore")

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from taas_checkmate import GpuHhLut
from taas_checkmate import SpikeGradBridge, T2SpikeGradTrajectory
from taas_checkmate import pst_linear_dim, t2spike_grad_pst_matrix

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01               # HH timestep (ms)
HH_STEPS = 500          # HH steps per token (5ms)
N_NEURONS = 3072        # HH neurons per layer
I_CLAMP = (0.0, 25.0)

# T3 config
N_OSC = 64              # T3 oscillators (drive_dim = 192)
T3_STEPS = 20           # T3 steps per token
T3_DT = 0.05            # T3 timestep
DRIVE_GAIN = 10.0
ALPHA = 1.0
FREQ_MIN = 0.5
FREQ_MAX = 3.0
COUPLING = (0.3, 0.15, 0.08)  # w12, w13, w23

# Training
LR_BRIDGE = 0.001
LR_DECODER = 0.001
N_EPOCHS = 50
SAVE_PATH = "snn_bridge_params.json"

PROMPTS = [
    "The quantum computer calculated the answer in seconds.",
    "Neural networks learn hierarchical representations of data.",
    "In the beginning, the universe was created.",
]


# ═══ Dataset: collect HH spikes + GELU targets ═══════════

print(f"{'='*68}")
print(f"  🌀 SNN × GPT-2 — SPIGRAD pipeline")
print(f"  {N_OSC} oscillators, {T3_STEPS} T3 steps, {N_EPOCHS} epochs")
print(f"{'='*68}")

print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config
PST_DIM = pst_linear_dim(N_OSC)
print(f"   {cfg.n_layer} layers, pst_dim={PST_DIM}")

# Collect dataset per layer: {spike_counts [n_tokens, 3072], target_h [n_tokens, 768]}
dataset = {}

for pi, prompt in enumerate(PROMPTS):
    print(f"\n📝 Prompt {pi+1}: \"{prompt}\"")
    tokens = tokenizer(prompt, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    
    h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
        torch.arange(seq_len).unsqueeze(0))
    h = model.transformer.drop(h)
    
    for layer_idx in range(cfg.n_layer):
        block = model.transformer.h[layer_idx]
        mlp = block.mlp
        
        with torch.no_grad():
            h = h + block.attn(block.ln_1(h))[0]
        
        ln_out = block.ln_2(h)
        pre = (ln_out.detach() @ mlp.c_fc.weight.detach() + mlp.c_fc.bias.detach())
        pre_np = pre.numpy()[0]  # [seq, 3072]
        
        # GELU target
        gelu_act = F.gelu(pre).numpy()[0]  # [seq, 3072]
        target_h = gelu_act @ mlp.c_proj.weight.detach().numpy() + mlp.c_proj.bias.detach().numpy()  # [seq, 768]
        
        # HH spikes (GPU) — reuse one HhLut per layer
        hh_spikes = np.zeros((seq_len, N_NEURONS), dtype=np.float32)
        hh = GpuHhLut(N_NEURONS, DT, pacemaking=False)
        for s in range(seq_len):
            I_cur = np.clip(pre_np[s] * 3.0 + 7.0, I_CLAMP[0], I_CLAMP[1])
            hh.set_I_all(I_cur.astype(np.float32))
            total = np.zeros(N_NEURONS, dtype=np.int32)
            v_prev = np.array(hh.v, dtype=np.float32)
            for _ in range(HH_STEPS):
                hh.step()
                v = np.array(hh.v, dtype=np.float32)
                total += ((v_prev >= 0) & (v < 0)).astype(np.int32)
                v_prev = v
            hh_spikes[s] = total.astype(np.float32)
            hh.reset()
        
        if layer_idx not in dataset:
            dataset[layer_idx] = {"spikes": [], "target": []}
        dataset[layer_idx]["spikes"].append(hh_spikes)  # [seq, 3072]
        dataset[layer_idx]["target"].append(target_h)   # [seq, 768]
        
        # Update h with MLP for next layer
        with torch.no_grad():
            h = h + mlp(ln_out)

# Concatenate all prompts
for li in range(cfg.n_layer):
    dataset[li]["spikes"] = np.concatenate(dataset[li]["spikes"], axis=0).astype(np.float32)
    dataset[li]["target"] = np.concatenate(dataset[li]["target"], axis=0).astype(np.float32)
    print(f"   Layer {li+1}: {dataset[li]['spikes'].shape[0]} tokens")

# ═══ Per-layer SPIGRAD training ═══════════════════════════

print(f"\n{'='*68}")
print(f"  🧠 SPIGRAD training per layer")
print(f"{'='*68}")

results = {}

for layer_idx in range(min(3, cfg.n_layer)):  # First 3 layers for PoC
    spikes_data = dataset[layer_idx]["spikes"]  # [N, 3072]
    target_data = dataset[layer_idx]["target"]   # [N, 768]
    n_tokens = spikes_data.shape[0]
    
    print(f"\n{'─'*50}")
    print(f"  Layer {layer_idx+1}: {n_tokens} tokens")
    print(f"{'─'*50}")
    
    # Create SpikeGradBridge
    bridge = SpikeGradBridge(
        spike_dim=N_NEURONS,
        n_osc=N_OSC,
        dt=T3_DT,
        drive_gain=DRIVE_GAIN,
        alpha=ALPHA,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        coupling_w12=COUPLING[0],
        coupling_w13=COUPLING[1],
        coupling_w23=COUPLING[2],
    )
    
    # Init decoder: PST → h_out [768]
    decode_w = np.random.randn(PST_DIM, 768).astype(np.float32) * 0.01
    decode_b = np.zeros(768, dtype=np.float32)
    
    loss_history = []
    t0 = time.perf_counter()
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        
        # Process each token independently (Mode A)
        for t in range(n_tokens):
            spike_t = spikes_data[t:t+1]  # [1, 3072]
            target_t = target_data[t]     # [768]
            
            # Tile spikes for all T3 steps: [T3_STEPS, spike_dim]
            # Rust forward() expects spikes[t * sd..(t+1) * sd] for each step
            spikes_tiled = np.tile(spike_t, (T3_STEPS, 1))  # [20, 3072]
            
            # Forward: spikes → bridge → T3 → PST
            pst_mat, traj = bridge.forward(spikes_tiled, T3_STEPS)
            # pst_mat: (T3_STEPS, PST_DIM) numpy 2D
            pst_final = pst_mat[-1]  # [PST_DIM] — последний шаг T3 (установившийся режим)
            
            # Decode: PST → h_out
            h_out = pst_final @ decode_w + decode_b  # [768]
            
            # Loss: MSE
            err = h_out - target_t  # [768]
            loss = 0.5 * np.mean(err ** 2)
            epoch_loss += loss
            
            # Per-step PST gradient для backward (2D: [T3_STEPS, PST_DIM])
            grad_h = err / 768.0  # dLoss/dh_out, [768]
            per_step_pst_2d = np.zeros((T3_STEPS, PST_DIM), dtype=np.float32)
            per_step_pst_2d[-1] = grad_h @ decode_w.T  # [768] @ [768, PST_DIM] = [PST_DIM]
            
            # Backward: BPTT through T3
            grad_pst_final = np.zeros(PST_DIM, dtype=np.float32)
            grad_w, grad_b = bridge.backward(grad_pst_final, traj, spikes_tiled, per_step_pst_2d)
            
            # Update bridge
            bridge.update(np.array(grad_w), np.array(grad_b), LR_BRIDGE)
            
            # Update decoder (numpy vectorized — БЕЗ Python loop!)
            err_scaled = err / 768.0
            decode_w -= LR_DECODER * np.outer(pst_final, err_scaled)  # [PST_DIM, 768]
            decode_b -= LR_DECODER * err_scaled  # [768]
        
        avg_loss = epoch_loss / n_tokens
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: loss={avg_loss:.6f}")
    
    t_elapsed = time.perf_counter() - t0
    improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100 if loss_history[0] > 0 else 0
    
    results[layer_idx] = {
        "loss_history": loss_history,
        "bridge_weights": bridge.weights(),
        "bridge_bias": bridge.bias(),
        "decoder_w": decode_w.tolist(),
        "decoder_b": decode_b.tolist(),
        "final_loss": float(loss_history[-1]),
        "improvement_pct": float(improvement),
        "time_s": round(t_elapsed, 2),
    }
    
    print(f"   ✅ Layer {layer_idx+1}: loss {loss_history[0]:.4f} → {loss_history[-1]:.4f} "
          f"({improvement:.0f}% reduction, {t_elapsed:.1f}s)")

# ═══ Summary ══════════════════════════════════════════════

print(f"\n{'='*68}")
print(f"  📊 SPIGRAD Training Summary")
print(f"{'='*68}")
print(f"\n  {'Layer':>6} {'Start Loss':>12} {'Final Loss':>12} {'Reduction':>10} {'Time':>8}")
print(f"  {'─'*48}")
for li in sorted(results.keys()):
    r = results[li]
    print(f"  {li+1:6d} {r['loss_history'][0]:12.6f} {r['final_loss']:12.6f} "
          f"{r['improvement_pct']:9.1f}% {r['time_s']:7.1f}s")

print(f"\n  💾 Saving to {SAVE_PATH}...")
with open(SAVE_PATH, "w") as f:
    json.dump({
        "version": "1.0",
        "model": "gpt2",
        "n_osc": N_OSC,
        "t3_steps": T3_STEPS,
        "n_layers_trained": len(results),
        "results": {str(k): v for k, v in results.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2, default=str)
print(f"   ✅ Saved!")

print(f"\n{'='*68}")
print(f"  🎉 SPIGRAD SNN × GPT-2 — Антон ❤️ Сардана")
print(f"{'='*68}")
