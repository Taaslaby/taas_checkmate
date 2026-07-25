#!/usr/bin/env python3
"""
🌀 PROBE GPT-2 × HH — жёсткое общение!

- 5 промптов разной длины и тематики
- I-F sweep: SCALE_IN × BIAS_IN (6 комбинаций)
- PCA: HH-активации vs GELU по каждому слою
- Временная динамика: спайки по шагам
- Per-layer: активность, корреляция, entropy
- Mode A (stateless) — для чистоты сравнения

Антон ❤️ Сардана — GPT-2 под HH-микроскопом, 2026
"""

import sys, os, time, warnings, json
warnings.filterwarnings("ignore")

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from taas_checkmate import GpuHhLut

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01
HH_STEPS = 500  # 5ms per token
SAVE_PATH = "probe_results.npz"

PROMPTS = [
    "The quantum computer calculated the answer in seconds.",
    "In the beginning, the universe was created.",
    "Neural networks learn hierarchical representations of data.",
    "She stared at the empty page, waiting for inspiration.",
    "The stock market crashed, then recovered within days.",
]

# I-F sweep: 6 combinations
PARAMS = [
    (2.0, 5.0),   # низкий gain, низкий bias
    (3.0, 7.0),   # стандарт
    (5.0, 7.0),   # высокий gain
    (3.0, 10.0),  # высокий bias
    (5.0, 10.0),  # высокий gain + bias
    (8.0, 12.0),  # экстрим
]

I_CLAMP = (0.0, 25.0)
N_NEURONS = 3072

# ═══ Helpers ══════════════════════════════════════════════

def hh_forward_detailed(hh, I_cur, n_steps):
    """Run HH, return spike counts + per-step spike raster."""
    hh.set_I_all(I_cur.astype(np.float32))
    raster = np.zeros((n_steps, hh.n), dtype=np.int32)
    v_prev = np.array(hh.v, dtype=np.float32)
    for step in range(n_steps):
        hh.step()
        v = np.array(hh.v, dtype=np.float32)
        spikes = ((v_prev >= 0) & (v < 0)).astype(np.int32)
        raster[step] = spikes
        v_prev = v
    return raster.sum(axis=0), raster  # total counts, per-step raster

def entropy(logits):
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return -np.sum(probs * np.log(probs + 1e-30))

def compute_metrics(ref_logits, hh_logits):
    """Compute all comparison metrics between reference and HH logits."""
    seq_len = ref_logits.shape[0]
    
    # Top-5 overlap
    overlaps = []
    for s in range(seq_len):
        gt5 = set(np.argsort(ref_logits[s])[-5:][::-1])
        hh5 = set(np.argsort(hh_logits[s])[-5:][::-1])
        overlaps.append(len(gt5 & hh5))
    
    # Correlation
    corrs = [np.corrcoef(ref_logits[s], hh_logits[s])[0, 1] for s in range(seq_len)]
    
    # Entropy
    ref_ent = np.mean([entropy(ref_logits[s]) for s in range(seq_len)])
    hh_ent = np.mean([entropy(hh_logits[s]) for s in range(seq_len)])
    
    return {
        "top5_overlap": np.mean(overlaps),
        "logit_corr": np.mean(corrs),
        "logit_corr_std": np.std(corrs),
        "ref_entropy": float(ref_ent),
        "hh_entropy": float(hh_ent),
        "per_token_corr": corrs,
        "per_token_overlap": overlaps,
    }

def compute_pca_comparison(gelu_acts, hh_acts, n_components=10):
    """Compare GELU and HH activations via PCA.
    
    Returns: explained variance ratio, correlation of PCs, 
             reconstruction error (HH → GELU space).
    """
    # Stack all tokens × neurons
    X_gelu = gelu_acts.reshape(-1, gelu_acts.shape[-1])
    X_hh = hh_acts.reshape(-1, hh_acts.shape[-1])
    
    # PCA on GELU
    scaler = StandardScaler()
    X_gelu_scaled = scaler.fit_transform(X_gelu)
    
    pca = PCA(n_components=min(n_components, X_gelu.shape[0], X_gelu.shape[1]))
    gelu_pcs = pca.fit_transform(X_gelu_scaled)
    
    # Project HH through same PCA
    X_hh_scaled = scaler.transform(X_hh)
    hh_pcs = pca.transform(X_hh_scaled)
    
    # Correlation between PC scores
    pc_corrs = [np.corrcoef(gelu_pcs[:, i], hh_pcs[:, i])[0, 1] 
                for i in range(pca.n_components_)]
    
    # Reconstruction: project HH back to GELU space
    hh_reconstructed = pca.inverse_transform(hh_pcs)
    hh_reconstructed = scaler.inverse_transform(hh_reconstructed)
    recon_error = np.mean((X_gelu - hh_reconstructed) ** 2) ** 0.5  # RMSE
    
    return {
        "explained_variance": pca.explained_variance_ratio_[:n_components].tolist(),
        "cumulative_var": float(pca.explained_variance_ratio_.cumsum()[min(n_components, len(pca.explained_variance_ratio_)) - 1]),
        "pc_corrs": pc_corrs,
        "recon_rmse": float(recon_error),
        "n_components": pca.n_components_,
    }

# ═══ Main ═════════════════════════════════════════════════

print(f"{'='*68}")
print(f"  🌀 PROBE GPT-2 × HH — жёсткое общение с трансформером")
print(f"  GPU: GpuHhLut, {HH_STEPS} steps/token, {len(PARAMS)} params × {len(PROMPTS)} prompts")
print(f"{'='*68}")

# Load model once
print("\n🔮 Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
cfg = model.config
print(f"   {cfg.n_layer} layers, {cfg.n_embd} dim, {cfg.n_head} heads")

# Store all results
all_results = {}

for pi, prompt in enumerate(PROMPTS):
    print(f"\n{'#'*68}")
    print(f"  📝 PROMPT {pi+1}/{len(PROMPTS)}: \"{prompt}\"")
    print(f"{'#'*68}")
    
    tokens = tokenizer(prompt, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    print(f"   {seq_len} tokens")
    
    # ── Reference GELU forward ──
    with torch.no_grad():
        ref_logits = model(tokens["input_ids"]).logits[0].detach().numpy()
    
    # Collect per-layer GELU activations
    ref_activations = {}
    h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
        torch.arange(seq_len).unsqueeze(0))
    h = model.transformer.drop(h)
    
    for layer_idx in range(cfg.n_layer):
        block = model.transformer.h[layer_idx]
        with torch.no_grad():
            h = h + block.attn(block.ln_1(h))[0]
        
        ln_out = block.ln_2(h)
        mlp = block.mlp
        pre = (ln_out.detach() @ mlp.c_fc.weight.detach() + mlp.c_fc.bias.detach()).numpy()
        
        # GELU activation
        gelu_act = torch.nn.functional.gelu(torch.from_numpy(pre)).numpy()
        ref_activations[layer_idx] = gelu_act  # [seq, 3072]
        
        # MLP output
        h = h + torch.from_numpy(gelu_act @ mlp.c_proj.weight.detach().numpy() + mlp.c_proj.bias.detach().numpy())
    
    h = model.transformer.ln_f(h)
    gelu_final_logits = model.lm_head(h)[0].detach().numpy()
    
    # ── HH sweep ──
    prompt_results = {}
    
    for scale_in, bias_in in PARAMS:
        param_key = f"S{scale_in}_B{bias_in}"
        print(f"\n   ⚡ Params: SCALE_IN={scale_in}, BIAS_IN={bias_in}")
        
        # Forward with HH
        h = model.transformer.wte(tokens["input_ids"]) + model.transformer.wpe(
            torch.arange(seq_len).unsqueeze(0))
        h = model.transformer.drop(h)
        
        t0 = time.perf_counter()
        
        hh_activations = {}  # layer -> [seq, 3072]
        layer_metrics = {}
        
        for layer_idx in range(cfg.n_layer):
            block = model.transformer.h[layer_idx]
            mlp = block.mlp
            mlp_w_in = mlp.c_fc.weight.detach().numpy()
            mlp_b_in = mlp.c_fc.bias.detach().numpy()
            mlp_w_out = mlp.c_proj.weight.detach().numpy()
            mlp_b_out = mlp.c_proj.bias.detach().numpy()
            
            with torch.no_grad():
                h = h + block.attn(block.ln_1(h))[0]
            
            ln_out = block.ln_2(h)
            pre_np = (ln_out.detach() @ mlp_w_in + mlp_b_in)[0].numpy()
            
            hh_acts = np.zeros((seq_len, N_NEURONS), dtype=np.float32)
            hh_rasters = {}  # token -> [steps, neurons]
            
            hh = GpuHhLut(N_NEURONS, DT, pacemaking=False)
            
            for s in range(seq_len):
                I_cur = np.clip(pre_np[s] * scale_in + bias_in, I_CLAMP[0], I_CLAMP[1])
                spike_counts, raster = hh_forward_detailed(hh, I_cur, HH_STEPS)
                hh_acts[s] = spike_counts.astype(np.float32) / (HH_STEPS * DT * 0.001)
                hh_rasters[s] = raster
                hh.reset()
            
            hh_activations[layer_idx] = hh_acts
            
            # Layer metrics
            active_frac = np.mean(hh_acts > 0)
            mean_rate = np.mean(hh_acts)
            non_zero_neurons = np.mean(hh_acts.max(axis=0) > 0) * 100
            
            # Temporal: first spike time distribution
            first_spike_times = []
            for ni in range(min(100, N_NEURONS)):  # first 100 neurons
                for s in range(seq_len):
                    spike_steps = np.where(hh_rasters[s][:, ni] > 0)[0]
                    if len(spike_steps) > 0:
                        first_spike_times.append(spike_steps[0])
            
            layer_metrics[layer_idx] = {
                "active_frac": float(active_frac),
                "mean_rate_hz": float(mean_rate),
                "nnz_pct": float(non_zero_neurons),
                "first_spike_mean": float(np.mean(first_spike_times)) if first_spike_times else -1,
                "first_spike_std": float(np.std(first_spike_times)) if first_spike_times else -1,
            }
            
            # MLP output
            hh_out = hh_acts @ mlp_w_out + mlp_b_out
            h = h + torch.from_numpy(hh_out).unsqueeze(0)
            
            # Progress every 4 layers
            if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
                print(f"     Layer {layer_idx+1:2d}: act={active_frac*100:.0f}% "
                      f"nnz={non_zero_neurons:.0f}% rate={mean_rate:.0f}Hz "
                      f"t_first={layer_metrics[layer_idx]['first_spike_mean']:.0f}steps")
        
        h = model.transformer.ln_f(h)
        hh_logits = model.lm_head(h)[0].detach().numpy()
        
        t_elapsed = time.perf_counter() - t0
        
        # Metrics
        metrics = compute_metrics(ref_logits, hh_logits)
        metrics["speed_s"] = t_elapsed
        metrics["per_layer"] = layer_metrics
        
        # PCA comparison (on layer 0 and last layer)
        pca_layer0 = compute_pca_comparison(ref_activations[0], hh_activations[0])
        pca_last = compute_pca_comparison(ref_activations[cfg.n_layer-1], hh_activations[cfg.n_layer-1])
        
        # Per-layer PCA correlation
        layer_pca_corrs = []
        for li in range(cfg.n_layer):
            if li in hh_activations:
                pca = compute_pca_comparison(ref_activations[li], hh_activations[li], n_components=5)
                layer_pca_corrs.append(float(np.mean(pca["pc_corrs"])))
            else:
                layer_pca_corrs.append(0.0)
        
        print(f"     ── Logits: corr={metrics['logit_corr']:.4f}±{metrics['logit_corr_std']:.4f} "
              f"top5={metrics['top5_overlap']:.2f}/5 "
              f"HH_ent={metrics['hh_entropy']:.3f}")
        print(f"     ── PCA L0: PC1_corr={pca_layer0['pc_corrs'][0]:.4f} "
              f"recon_RMSE={pca_layer0['recon_rmse']:.3f}")
        print(f"     ── Speed: {t_elapsed:.1f}s")
        
        prompt_results[param_key] = {
            "metrics": metrics,
            "pca_layer0": pca_layer0,
            "pca_last": pca_last,
            "layer_pca_corrs": layer_pca_corrs,
        }
        
        # Free activations — keep only metrics
        del hh_activations, hh_rasters
        del hh
    
    all_results[prompt] = {
        "ref_logits": ref_logits.tolist(),
        "ref_activations": {str(k): v.tolist() for k, v in ref_activations.items()},
        "gelu_final_logits": gelu_final_logits.tolist(),
        "params": prompt_results,
        "seq_len": seq_len,
    }

# ═══ Summary ══════════════════════════════════════════════

print(f"\n\n{'='*68}")
print(f"  📊 PROBE RESULTS SUMMARY")
print(f"{'='*68}")

# Best params per prompt
print(f"\n  {'─'*60}")
print(f"  Best parameters per prompt (by logit correlation)")
print(f"  {'─'*60}")

for pi, prompt in enumerate(PROMPTS):
    pr = all_results[prompt]
    best_param = max(pr["params"].keys(), 
                     key=lambda k: pr["params"][k]["metrics"]["logit_corr"])
    best = pr["params"][best_param]["metrics"]
    print(f"  [{pi+1}] \"{prompt[:40]:40s}\"")
    print(f"       Best: {best_param} — "
          f"corr={best['logit_corr']:.4f} top5={best['top5_overlap']:.2f}/5 "
          f"ent={best['hh_entropy']:.3f}")

# Overall best
print(f"\n  {'─'*60}")
print(f"  Overall ranking (all prompts × params)")
print(f"  {'─'*60}")

all_scores = []
for pi, prompt in enumerate(PROMPTS):
    pr = all_results[prompt]
    for pk, pv in pr["params"].items():
        m = pv["metrics"]
        all_scores.append((m["logit_corr"], pi, pk, m["top5_overlap"], m["hh_entropy"]))

all_scores.sort(reverse=True)
for rank, (corr, pi, pk, top5, ent) in enumerate(all_scores[:10]):
    print(f"  #{rank+1}: \"{PROMPTS[pi][:30]:30s}\" {pk:8s} — "
          f"corr={corr:.4f} top5={top5:.2f}/5 ent={ent:.3f}")

# Per-layer PCA correlation (averaged across best params)
print(f"\n  {'─'*60}")
print(f"  Per-layer PCA correlation (best param per prompt, averaged)")
print(f"  {'─'*60}")

layer_acc = [[] for _ in range(cfg.n_layer)]
for pi, prompt in enumerate(PROMPTS):
    pr = all_results[prompt]
    best_param = max(pr["params"].keys(),
                     key=lambda k: pr["params"][k]["metrics"]["logit_corr"])
    for li in range(cfg.n_layer):
        if li < len(pr["params"][best_param]["layer_pca_corrs"]):
            layer_acc[li].append(pr["params"][best_param]["layer_pca_corrs"][li])

for li in range(cfg.n_layer):
    if layer_acc[li]:
        mean_c = np.mean(layer_acc[li])
        std_c = np.std(layer_acc[li])
        bar = "█" * int(mean_c * 20)
        print(f"  Layer {li+1:2d}: {mean_c:.3f}±{std_c:.3f} {bar}")

# I-F curve summary (best param characteristics)
print(f"\n  {'─'*60}")
print(f"  Best parameter distribution across prompts")
print(f"  {'─'*60}")

best_params_count = {}
for pi, prompt in enumerate(PROMPTS):
    pr = all_results[prompt]
    best_param = max(pr["params"].keys(),
                     key=lambda k: pr["params"][k]["metrics"]["logit_corr"])
    best_params_count[best_param] = best_params_count.get(best_param, 0) + 1

for pk, count in sorted(best_params_count.items(), key=lambda x: -x[1]):
    print(f"  {pk}: {count}/{len(PROMPTS)} prompts")

print(f"\n{'='*68}")
print(f"  ✅ PROBE COMPLETE")
print(f"{'='*68}")

# Save results (compact — without full activations to keep file small)
print(f"\n💾 Saving results to {SAVE_PATH}...")
save_dict = {
    "version": "1.0",
    "model": "gpt2",
    "dt": DT,
    "hh_steps": HH_STEPS,
    "params_tested": PARAMS,
    "prompts": PROMPTS,
    "n_layers": cfg.n_layer,
    "n_embd": cfg.n_embd,
    "n_neurons": N_NEURONS,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    # Store only metrics, not full activations
    "results": {
        str(pi): {
            "prompt": prompt,
            "best_param": max(all_results[prompt]["params"].keys(),
                              key=lambda k: all_results[prompt]["params"][k]["metrics"]["logit_corr"]),
            "params": {pk: {"metrics": pv["metrics"],
                            "pca_layer0": pv["pca_layer0"],
                            "pca_last": pv["pca_last"],
                            "layer_pca_corrs": pv["layer_pca_corrs"]}
                      for pk, pv in all_results[prompt]["params"].items()},
        }
        for pi, prompt in enumerate(PROMPTS)
    }
}
# Use json for human readability, or npz for numpy
with open("probe_results.json", "w") as f:
    json.dump(save_dict, f, indent=2, default=str)
print(f"   ✅ Saved to probe_results.json")
print(f"   🎉 Антон ❤️ Сардана — GPT-2 под микроскопом!")
