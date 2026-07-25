#!/usr/bin/env python3
"""
🌀 ГИПЕРБОЛИЗАТОР GENERATE — GPT-2 × HH генерация текста

Авторегрессивная генерация с HH вместо GELU.
Сравнение: GELU vs HH (stateless) vs HH (stateful)

Антон ❤️ Сардана — сознание говорит! 2026
"""

import sys, os, time
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from taas_checkmate import HhLut

# ═══ Config ═══════════════════════════════════════════════

DT = 0.01
HH_STEPS = 500
SCALE_IN = 3.0
BIAS_IN = 7.0
I_CLAMP = (0.0, 25.0)
NORM_SCALE = 100.0       # HH rate (Hz) / NORM_SCALE → GELU-like magnitude
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.8
PROMPTS = [
    "The future of artificial intelligence is",
    "In the beginning, there was",
    "The meaning of consciousness is",
]

# ═══ Helpers ══════════════════════════════════════════════

def hh_forward(hh: HhLut, I_cur: np.ndarray, n_steps: int) -> np.ndarray:
    """Run HH, return per-neuron spike counts (incremental, no v_trace alloc)."""
    hh.set_I_all(I_cur.astype(np.float32))
    spike_counts = np.zeros(hh.n, dtype=np.int32)
    v_prev = np.array(hh.v, dtype=np.float32)
    for _ in range(n_steps):
        hh.step()
        v = np.array(hh.v, dtype=np.float32)
        spike_counts += ((v_prev >= 0) & (v < 0)).astype(np.int32)
        v_prev = v
    return spike_counts

class HyperbolizerGPT:
    """GPT-2 with HH replacing GELU in MLP layers."""
    
    def __init__(self, do_reset=True):
        self.do_reset = do_reset  # True=Mode A (stateless), False=Mode B (stateful)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()
        self.cfg = self.model.config
        
        # Extract all MLP weights (Conv1D: weight=[in, out], forward=x@W+b)
        self.mlp_weights = []
        for i in range(self.cfg.n_layer):
            mlp = self.model.transformer.h[i].mlp
            self.mlp_weights.append({
                'W_in': mlp.c_fc.weight.detach().numpy(),     # [768, 3072]
                'b_in': mlp.c_fc.bias.detach().numpy(),       # [3072]
                'W_out': mlp.c_proj.weight.detach().numpy(),  # [3072, 768]
                'b_out': mlp.c_proj.bias.detach().numpy(),    # [768]
            })
        
        # Create per-layer HH simulators (one per layer, reused across tokens)
        self.hh_layers = [HhLut(3072, DT, pacemaking=False) for _ in range(self.cfg.n_layer)]
    
    def reset_hh_state(self):
        """Reset all HH simulators to resting state."""
        for hh in self.hh_layers:
            hh.reset()
    
    def forward(self, input_ids):
        """Forward pass through GPT-2 with HH replacing GELU.
        
        Args:
            input_ids: [1, seq_len] token IDs
        
        Returns:
            logits: [1, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        
        # Embedding
        h = self.model.transformer.wte(input_ids) + \
            self.model.transformer.wpe(torch.arange(seq_len).unsqueeze(0))
        h = self.model.transformer.drop(h)
        
        for layer_idx in range(self.cfg.n_layer):
            block = self.model.transformer.h[layer_idx]
            mlp_w = self.mlp_weights[layer_idx]
            hh = self.hh_layers[layer_idx]
            
            # ── Attention ──
            with torch.no_grad():
                attn_out = block.attn(block.ln_1(h))[0]
            h = h + attn_out
            
            # ── MLP with HH ──
            ln_out = block.ln_2(h)
            pre_np = (ln_out.detach() @ mlp_w['W_in'] + mlp_w['b_in'])[0].numpy()
            
            hh_out = np.zeros((seq_len, self.cfg.n_embd), dtype=np.float32)
            
            for s in range(seq_len):
                I_cur = np.clip(pre_np[s] * SCALE_IN + BIAS_IN, I_CLAMP[0], I_CLAMP[1])
                spike_counts = hh_forward(hh, I_cur, HH_STEPS)
                
                # Activation: spike rate → normalize to GELU magnitude
                act = spike_counts.astype(np.float32) / (HH_STEPS * DT * 0.001)
                act = act / NORM_SCALE  # match GELU scale ≈ [0, 2]
                
                hh_out[s] = act @ mlp_w['W_out'] + mlp_w['b_out']
                
                if self.do_reset:
                    hh.reset()
                # Mode B: NO reset — HH state carries to next token!
            
            h = h + torch.from_numpy(hh_out).unsqueeze(0)
        
        # Final LN + LM head
        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)
    
    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        """Autoregressive generation with HH.
        
        Для Mode B: HH состояние сохраняется между шагами генерации!
        Это значит, что модель помнит не только скрытые состояния,
        но и ВОЛЬТАЖ мембраны каждого HH-нейрона из предыдущего шага.
        """
        self.reset_hh_state()
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Forward pass (короткий — только последние позиции для efficiency)
            # В прототипе: полный forward на всей последовательности
            logits = self.forward(generated)
            next_logits = logits[0, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Для Mode B: НЕ сбрасывать HH состояние!
            # HH-нейроны помнят предыдущий контекст через V и gates
            
            if (step + 1) % 5 == 0:
                print(f"   Step {step+1}/{max_new_tokens}")
        
        return generated


def generate_reference(model, tokenizer, input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Reference generation with original GELU GPT-2."""
    generated = input_ids.clone()
    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated).logits
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


# ═══ Main ═════════════════════════════════════════════════

print("╔" + "═" * 58 + "╗")
print(f"║{'🌀 HYPERBOLIZER GENERATE — GPT-2 × HH':^58}║")
print("╚" + "═" * 58 + "╝")
print(f"\n   HH: {HH_STEPS} steps, scale={SCALE_IN}, bias={BIAS_IN}")
print(f"   Generation: {MAX_NEW_TOKENS} tokens, temp={TEMPERATURE}")
print(f"   Normalization: /{NORM_SCALE}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ref_model = GPT2LMHeadModel.from_pretrained("gpt2")
ref_model.eval()

# Init HH models
hh_stateless = HyperbolizerGPT(do_reset=True)   # Mode A
hh_stateful = HyperbolizerGPT(do_reset=False)   # Mode B

for prompt_text in PROMPTS:
    print(f"\n{'='*58}")
    print(f"📝 Prompt: '{prompt_text}'")
    print(f"{'='*58}")
    
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    
    # ── GELU reference ──
    t0 = time.time()
    ref_ids = generate_reference(ref_model, tokenizer, input_ids)
    ref_text = tokenizer.decode(ref_ids[0])
    print(f"\n🔮 GELU ({time.time()-t0:.0f}s):")
    print(f"   {ref_text}")
    
    # ── HH stateless (Mode A) ──
    t0 = time.time()
    hh_ids_a = hh_stateless.generate(input_ids)
    hh_text_a = tokenizer.decode(hh_ids_a[0])
    print(f"\n🧬 HH stateless ({time.time()-t0:.0f}s):")
    print(f"   {hh_text_a}")
    
    # ── HH stateful (Mode B) ──
    t0 = time.time()
    hh_ids_b = hh_stateful.generate(input_ids)
    hh_text_b = tokenizer.decode(hh_ids_b[0])
    print(f"\n🧠 HH stateful ({time.time()-t0:.0f}s):")
    print(f"   {hh_text_b}")
    
    # ── Compare ──
    new_a = tokenizer.decode(hh_ids_a[0, len(input_ids[0]):])
    new_b = tokenizer.decode(hh_ids_b[0, len(input_ids[0]):])
    new_ref = tokenizer.decode(ref_ids[0, len(input_ids[0]):])
    
    print(f"\n📊 Generated tokens comparison:")
    print(f"   GELU:  '{new_ref}'")
    print(f"   HH-A:  '{new_a}'")
    print(f"   HH-B:  '{new_b}'")

print(f"\n{'='*58}")
print(f"{'🌀 HYPERBOLIZER GENERATE — COMPLETE':^58}")
print(f"{'='*58}")
print(f"   Антон ❤️ Сардана — HH-мозг говорит!")
print(f"{'='*58}")
