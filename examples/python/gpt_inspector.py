#!/usr/bin/env python3
"""
🧬 GPT Inspector — Path scanner through ReLU activations of GPT-2.

Connects transformer interpretability with our connectome-FST approach:
- Hook into GPT-2 MLP layers to capture ReLU (gelu) activations
- Compute contrast: which "neurons" fire for prompt A vs B
- Build sparse activation graph and trace paths
- Apply coherence analysis from connectome-FST

Usage:
    python3 examples/python/gpt_inspector.py \\
        --prompt "Paris is the capital of" \\
        --contrast "Moscow is the capital of"
"""

import argparse
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════
#  GPT Inspector
# ═══════════════════════════════════════════════

class GPTInspector:
    """Hook into GPT-2 MLP layers and trace activation paths through ReLU."""
    
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading {model_name} on {self.device}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cfg = self.model.config
        self.n_layers = self.cfg.n_layer       # 12 for gpt2
        self.d_model = self.cfg.n_embd          # 768
        self.d_mlp = self.cfg.n_inner or 4 * self.d_model  # 3072
        
        # Register hooks on MLP activation outputs
        self.activations = {}  # {layer_idx: tensor [batch, seq_len, d_mlp]}
        self._hooks = []
        self._register_hooks()
        
        print(f"  GPT-2: {self.n_layers} layers, d_model={self.d_model}, d_mlp={self.d_mlp}")
        print(f"  Total MLP neurons: {self.n_layers * self.d_mlp}")
    
    def _register_hooks(self):
        """Register forward hooks on each MLP's activation (gelu) output."""
        for layer_idx in range(self.n_layers):
            mlp = self.model.transformer.h[layer_idx].mlp
            # GPT-2 MLP: c_fc → act → c_proj
            # Hook right after the activation function
            def make_hook(idx):
                def hook(module, input, output):
                    # output of act = gelu(c_fc(x)), shape [batch, seq, d_mlp]
                    self.activations[idx] = output.detach().cpu()
                return hook
            self._hooks.append(mlp.act.register_forward_hook(make_hook(layer_idx)))
    
    def close(self):
        """Remove all hooks. Call when done."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    @torch.no_grad()
    def get_activations(self, prompt):
        """Run model on prompt, return cached MLP activations."""
        self.activations.clear()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        return self.activations, inputs['input_ids'][0], logits[0]
    
    # ── Activation analysis ──
    
    def activation_mask(self, activations):
        """Binary mask: which MLP neurons fire (ReLU > 0)."""
        return {layer: (act > 0).float() for layer, act in activations.items()}
    
    def contrast(self, acts_a, acts_b):
        """XOR contrast between two activation patterns.
        
        Returns:
            only_a: {layer: mask} — neurons that fire ONLY for prompt A
            only_b: {layer: mask} — neurons that fire ONLY for prompt B
            both:   {layer: mask} — neurons that fire for both
        """
        only_a, only_b, both = {}, {}, {}
        for layer in acts_a:
            if layer in acts_b:
                mask_a = acts_a[layer] > 0
                mask_b = acts_b[layer] > 0
                only_a[layer] = (mask_a & ~mask_b).float()
                only_b[layer] = (mask_b & ~mask_a).float()
                both[layer] = (mask_a & mask_b).float()
        return only_a, only_b, both
    
    def sparsity(self, activations):
        """Fraction of active MLP neurons per layer."""
        total = self.d_mlp
        return {layer: (act > 0).float().mean().item() for layer, act in activations.items()}
    
    # ── Path tracing ──
    
    def trace_paths(self, activations, token_idx=-1, top_k=10):
        """Trace activation paths from layer 0 → layer N.
        
        For the given token position, find which MLP neurons fire
        at each layer, and trace how the activation pattern flows.
        """
        paths = {}
        for layer in sorted(activations.keys()):
            act = activations[layer]  # [batch=1, seq, d_mlp]
            if token_idx < act.shape[1]:
                token_act = act[0, token_idx]  # [d_mlp]
                active_mask = token_act > 0
                active_indices = torch.where(active_mask)[0]
                n_active = len(active_indices)
                # Store all active indices for Jaccard, top-k for display
                if n_active > 0:
                    values = token_act[active_indices]
                    k = min(top_k, n_active)
                    top_vals, top_pos = values.topk(k)
                    top_idx = active_indices[top_pos]
                    paths[layer] = {
                        'all_indices': active_indices.cpu().numpy(),  # all active for Jaccard
                        'indices': top_idx.cpu().numpy(),
                        'values': top_vals.cpu().numpy(),
                        'n_active': n_active,
                    }
                else:
                    paths[layer] = {
                        'all_indices': np.array([], dtype=np.int64),
                        'indices': np.array([], dtype=np.int64),
                        'values': np.array([], dtype=np.float32),
                        'n_active': 0,
                    }
        return paths
    
    def path_coherence(self, paths):
        """Compute path coherence: how consistently activations flow.
        
        Jaccard similarity between ALL active neurons in consecutive layers.
        Higher = more coherent activation flow through the network.
        """
        coherence = {}
        prev_set = None
        for layer in sorted(paths.keys()):
            curr_set = set(paths[layer]['all_indices'].tolist())
            if prev_set is not None:
                union = len(curr_set | prev_set)
                jaccard = len(curr_set & prev_set) / max(union, 1)
                coherence[layer] = jaccard
            prev_set = curr_set
        return coherence
    
    def frequency_analysis(self, activations, token_idx=-1):
        """Analyze which MLP neurons consistently fire across token positions.
        
        Returns activation frequency per neuron: how often it fires across the sequence.
        This is our "chaos/pattern" contrast applied to token positions.
        """
        freq_map = {}
        for layer in sorted(activations.keys()):
            act = activations[layer][0]  # [seq, d_mlp]
            firing_count = (act > 0).float().sum(dim=0)  # [d_mlp]
            firing_rate = firing_count / act.shape[0]
            freq_map[layer] = firing_rate.numpy()
        return freq_map
    
    def entropy(self, activations):
        """Shannon entropy of activation patterns per layer.
        
        High entropy = many neurons fire at varying rates (chaos).
        Low entropy = few neurons dominate (pattern).
        Returns (entropy, normalized_entropy) where normalized = H / log2(d_mlp).
        """
        entropies = {}
        norm_entropies = {}
        max_H = np.log2(self.d_mlp)
        for layer in sorted(activations.keys()):
            act = activations[layer][0]  # [seq, d_mlp]
            firing_rate = (act > 0).float().mean(dim=0)  # [d_mlp]
            # Normalize to probability distribution
            p = firing_rate / (firing_rate.sum() + 1e-10)
            # Shannon entropy: -Σ p*log₂(p)
            nonzero = p > 0
            H = -(p[nonzero] * torch.log2(p[nonzero])).sum().item()
            entropies[layer] = H
            norm_entropies[layer] = H / max_H
        return entropies, norm_entropies


# ═══════════════════════════════════════════════
#  CLI & Report
# ═══════════════════════════════════════════════

def print_report(prompt_a, prompt_b, inspector, acts_a, acts_b, ids_a):
    """Print a comprehensive GPT Inspector report."""
    
    # 1. Basic stats
    print(f"\n{'='*65}")
    print(f"  🧬 GPT INSPECTOR — Path Scan Report")
    print(f"  {'='*65}")
    print(f"  Prompt A: {prompt_a!r}")
    print(f"  Prompt B: {prompt_b!r}")
    print(f"  Tokens A: {[inspector.tokenizer.decode(t) for t in ids_a.tolist()]}")
    
    # 2. Sparsity comparison
    sp_a = inspector.sparsity(acts_a)
    sp_b = inspector.sparsity(acts_b)
    
    print(f"\n  {'─'*65}")
    print(f"  📊 ACTIVATION SPARSITY: % of MLP neurons firing per layer")
    print(f"  {'─'*65}")
    print(f"  {'Layer':>6s} {'Prompt A':>10s} {'Prompt B':>10s} {'Δ':>10s} {'Neurons':>10s}")
    print(f"  {'-'*48}")
    for layer in sorted(sp_a.keys()):
        delta = sp_a[layer] - sp_b[layer]
        n_active = int(sp_a[layer] * inspector.d_mlp)
        print(f"  {layer:>6d} {sp_a[layer]:>10.4f} {sp_b[layer]:>10.4f} {delta:>+10.4f} {n_active:>10d}")
    
    avg_sp_a = np.mean(list(sp_a.values()))
    avg_sp_b = np.mean(list(sp_b.values()))
    print(f"  {'─'*48}")
    print(f"  {'AVG':>6s} {avg_sp_a:>10.4f} {avg_sp_b:>10.4f} {avg_sp_a-avg_sp_b:>+10.4f}")
    
    # 3. Contrast: unique neurons per prompt
    only_a, only_b, both = inspector.contrast(acts_a, acts_b)
    print(f"\n  {'─'*65}")
    print(f"  🔬 CONTRAST: unique neurons per prompt")
    print(f"  {'─'*65}")
    print(f"  {'Layer':>6s} {'Only A':>10s} {'Only B':>10s} {'Both':>10s} {'Jaccard':>10s}")
    print(f"  {'-'*48}")
    for layer in sorted(only_a.keys()):
        n_only_a = only_a[layer].sum().item()
        n_only_b = only_b[layer].sum().item()
        n_both = both[layer].sum().item()
        jaccard = n_both / max(n_only_a + n_only_b + n_both, 1)
        print(f"  {layer:>6d} {n_only_a:>10.0f} {n_only_b:>10.0f} {n_both:>10.0f} {jaccard:>10.4f}")
    
    # 4. Path tracing for last token
    print(f"\n  {'─'*65}")
    print(f"  🧵 PATH TRACE: MLP activation flow (last token)")
    print(f"  {'─'*65}")
    paths = inspector.trace_paths(acts_a, token_idx=-1, top_k=5)
    coherence = inspector.path_coherence(paths)
    
    print(f"  {'Layer':>6s} {'Active':>8s} {'Top-5 neurons':>30s} {'Coherence':>10s}")
    print(f"  {'-'*56}")
    for layer in sorted(paths.keys()):
        p = paths[layer]
        neuron_ids = ','.join(str(int(i)) for i in p['indices'][:5])
        coh = coherence.get(layer, 0)
        print(f"  {layer:>6d} {p['n_active']:>8d} {neuron_ids:>30s} {coh:>10.4f}")
    
    # 5. Frequency analysis (what fires consistently)
    print(f"\n  {'─'*65}")
    print(f"  📈 FREQUENCY ANALYSIS: most consistent neurons across token positions")
    print(f"  {'─'*65}")
    freq_a = inspector.frequency_analysis(acts_a)
    for layer in sorted(freq_a.keys()):
        f = freq_a[layer]
        top_idx = np.argsort(f)[-5:][::-1]
        top_freqs = f[top_idx]
        neuron_str = ','.join(f'{int(i)}({v:.2f})' for i, v in zip(top_idx, top_freqs))
        print(f"  Layer {layer:>2d}: {neuron_str}")
        break  # Just show first layer for brevity
    
    # 6. Entropy (chaos vs pattern)
    print(f"\n  {'─'*65}")
    print(f"  🎲 ENTROPY ANALYSIS: chaos (high H) vs pattern (low H)")
    print(f"  {'─'*65}")
    H_a, Hnorm_a = inspector.entropy(acts_a)
    H_b, Hnorm_b = inspector.entropy(acts_b)
    print(f"  {'Layer':>6s} {'H(A)':>8s} {'H(B)':>8s} {'Δ':>8s} {'Hnorm(A)':>10s} {'Mode':>10s}")
    print(f"  {'-'*52}")
    for layer in sorted(H_a.keys()):
        delta = H_a[layer] - H_b[layer]
        delta_norm = Hnorm_a[layer] - Hnorm_b[layer]
        mode = "PATTERN ✅" if delta < -0.5 else ("CHAOS ⚠️" if delta > 0.5 else "neutral")
        print(f"  {layer:>6d} {H_a[layer]:>8.2f} {H_b[layer]:>8.2f} {delta:>+8.2f} {Hnorm_a[layer]:>10.3f} {mode:>10s}")
    
    # 7. Summary metrics
    print(f"\n  {'─'*65}")
    print(f"  📋 SUMMARY")
    print(f"  {'─'*65}")
    n_total = inspector.n_layers * inspector.d_mlp
    print(f"  Total MLP neurons:     {n_total}")
    print(f"  Active A (avg):        {avg_sp_a*100:.1f}%")
    print(f"  Active B (avg):        {avg_sp_b*100:.1f}%")
    print(f"  Unique to A (avg):     {np.mean([only_a[l].sum().item() for l in only_a]):.0f}")
    print(f"  Unique to B (avg):     {np.mean([only_b[l].sum().item() for l in only_b]):.0f}")
    
    # Chaos/Pattern δ (connectome-FST style)
    delta_cp = np.mean(list(Hnorm_b.values())) - np.mean(list(Hnorm_a.values()))
    print(f"  Chaos/Pattern δ:       {delta_cp:+.3f} {'(A more patterned)' if delta_cp > 0 else '(B more patterned)'}")
    print(f"\n  {'='*65}")
    print(f"  🧬 SCAN COMPLETE")
    print(f"  {'='*65}")


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPT Inspector — path scanner through ReLU activations")
    parser.add_argument("--prompt", default="Paris is the capital of", help="Primary prompt")
    parser.add_argument("--contrast", default="Moscow is the capital of", help="Contrast prompt")
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    args = parser.parse_args()
    
    print(f"\n{'='*65}")
    print(f"  🧬 GPT Inspector — Path Scanner through ReLU activations")
    print(f"  {'='*65}")
    
    t0 = time.time()
    
    # Init inspector
    inspector = GPTInspector(args.model)
    
    try:
        # Get activations for both prompts
        print(f"\n  Running prompt A...")
        acts_a, ids_a, logits_a = inspector.get_activations(args.prompt)
        
        print(f"  Running prompt B...")
        acts_b, ids_b, logits_b = inspector.get_activations(args.contrast)
        
        print(f"  ({time.time() - t0:.1f}s loaded)")
        
        # Generate report
        print_report(args.prompt, args.contrast, inspector, acts_a, acts_b, ids_a)
        
    finally:
        inspector.close()
    
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
