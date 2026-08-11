#!/usr/bin/env python
"""SpikeGPT-216M — ground-truth reference (parity target for the Rust engine).

Faithful replica of ridgerchu/SpikeGPT `src/model_run.py` (RWKV_RNN inference
form, FLOAT_MODE=fp32, RUN_DEVICE=cpu) + `run.py` weight preprocessing.

Modes:
  convert   — .pth -> target/weights/weights.bin (f32 blob) + manifest.json
              (the exact preprocessed tensors the Rust engine must load)
  ref       — run the reference forward over a prompt, dump
              target/weights/logits_ref.bin (T x VOCAB f32) + tokens.txt
  gen       — autoregressive generation (temperature/top-p), like run.py

Usage:
  python resources/examples/spikegpt_ref.py convert <model.pth> <outdir>
  python resources/examples/spikegpt_ref.py ref <outdir> [--prompt "text"] [--tokens "1 2 3"]
  python resources/examples/spikegpt_ref.py gen <outdir> [--prompt "text"] [--length 64] [--seed 0]

All math mirrors model_run.py line-by-line: time-decay = -exp(w),
att.output / ffn.value rescaled by 2^(layer//6), x/2 every 6 layers,
LIFNode(tau=2, hard reset v_reset=0) applied after att and ffn.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

N_LAYER = 18
N_EMBD = 768
VOCAB = 50277
RESCALE = 6          # RWKV_RESCALE_LAYER
CTX_LEN = 1024


# ---------------------------------------------------------------- preprocessing
def load_preprocessed(model_path):
    """Load .pth and apply model_run.py preprocessing exactly."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    w = {}
    for k, v in ckpt.items():
        if not isinstance(v, torch.Tensor):
            continue
        v = v.float()
        if ".time_" in k:
            v = v.squeeze()
        if ".time_decay" in k:
            v = -torch.exp(v)
        if ".att.output.weight" in k:
            v = v / (2.0 ** (int(k.split(".")[1]) // RESCALE))
        if ".ffn.value.weight" in k:
            v = v / (2.0 ** (int(k.split(".")[1]) // RESCALE))
        w[k] = v.contiguous()
    return w


def dump_for_rust(w, outdir):
    """weights.bin (f32 little-endian, name order) + manifest.json."""
    os.makedirs(outdir, exist_ok=True)
    manifest = {"n_embd": N_EMBD, "n_layer": N_LAYER, "vocab": VOCAB, "tensors": {}}
    buf = bytearray()
    for name in sorted(w.keys()):
        v = w[name]
        off = len(buf)
        buf += v.numpy().astype(np.float32).tobytes()
        manifest["tensors"][name] = {"offset": off, "shape": list(v.shape)}
    with open(os.path.join(outdir, "weights.bin"), "wb") as f:
        f.write(buf)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=0)
    print(f"wrote weights.bin ({len(buf)/1e6:.0f} MB) + manifest.json "
          f"({len(manifest['tensors'])} tensors) -> {outdir}")


# ---------------------------------------------------------------- the forward
def layer_norm(x, w, b):
    return F.layer_norm(x, (N_EMBD,), weight=w, bias=b, eps=1e-5)


def lif_forward(v, x, tau=2.0, v_threshold=1.0):
    """spikingjelly LIFNode(tau=2, decay_input=True, v_reset=0.0) — one step."""
    v = v + (x - v) / tau
    spike = (v >= v_threshold).float()
    v = v * (1.0 - spike)          # hard reset to v_reset = 0
    return v, spike


class Runner:
    """Stateful per-token forward, B=1, replicating RWKV_RNN.forward."""

    def __init__(self, w):
        self.w = w
        self.state = None          # n_layer*5 x n_embd
        self.lif_v1 = None         # n_layer x n_embd
        self.lif_v2 = None

    def reset(self):
        self.state = torch.zeros(N_LAYER * 5, N_EMBD)
        for i in range(N_LAYER):
            self.state[5 * i + 4] -= 1e30      # pp starts at -1e30
        self.lif_v1 = torch.zeros(N_LAYER, N_EMBD)
        self.lif_v2 = torch.zeros(N_LAYER, N_EMBD)

    def step(self, tok, logits=False):
        w, st = self.w, self.state
        x = w["emb.weight"][tok].clone()
        for i in range(N_LAYER):
            if i == 0:
                x = layer_norm(x, w["blocks.0.ln0.weight"], w["blocks.0.ln0.bias"])
            # ---- SA (attention) ----
            xx = st[5 * i + 1]
            xin = layer_norm(x, w[f"blocks.{i}.ln1.weight"], w[f"blocks.{i}.ln1.bias"])
            tm_k = w[f"blocks.{i}.att.time_mix_k"]
            tm_v = w[f"blocks.{i}.att.time_mix_v"]
            tm_r = w[f"blocks.{i}.att.time_mix_r"]
            xk = xin * tm_k + xx * (1.0 - tm_k)
            xv = xin * tm_v + xx * (1.0 - tm_v)
            xr = xin * tm_r + xx * (1.0 - tm_r)
            st[5 * i + 1] = xin
            r = torch.sigmoid(w[f"blocks.{i}.att.receptance.weight"] @ xr)
            k = w[f"blocks.{i}.att.key.weight"] @ xk
            v = w[f"blocks.{i}.att.value.weight"] @ xv
            aa, bb, pp = st[5 * i + 2], st[5 * i + 3], st[5 * i + 4]
            tf = w[f"blocks.{i}.att.time_first"]
            td = w[f"blocks.{i}.att.time_decay"]
            ww = tf + k
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v
            b = e1 * bb + e2
            ww2 = pp + td
            p2 = torch.maximum(ww2, k)
            e1b = torch.exp(ww2 - p2)
            e2b = torch.exp(k - p2)
            st[5 * i + 2] = e1b * aa + e2b * v
            st[5 * i + 3] = e1b * bb + e2b
            st[5 * i + 4] = p2
            wkv = a / b
            att_out = w[f"blocks.{i}.att.output.weight"] @ (r * wkv)
            self.lif_v1[i], spike1 = lif_forward(self.lif_v1[i], att_out)
            x = x + spike1
            # ---- FF (channel mix) ----
            xx2 = st[5 * i + 0]
            xin2 = layer_norm(x, w[f"blocks.{i}.ln2.weight"], w[f"blocks.{i}.ln2.bias"])
            tm_k2 = w[f"blocks.{i}.ffn.time_mix_k"]
            tm_r2 = w[f"blocks.{i}.ffn.time_mix_r"]
            xk2 = xin2 * tm_k2 + xx2 * (1.0 - tm_k2)
            xr2 = xin2 * tm_r2 + xx2 * (1.0 - tm_r2)
            st[5 * i + 0] = xin2
            r2 = torch.sigmoid(w[f"blocks.{i}.ffn.receptance.weight"] @ xr2)
            kk = w[f"blocks.{i}.ffn.key.weight"] @ xk2
            kk = torch.square(torch.relu(kk))
            kv = w[f"blocks.{i}.ffn.value.weight"] @ kk
            ffn_out = r2 * kv
            self.lif_v2[i], spike2 = lif_forward(self.lif_v2[i], ffn_out)
            x = x + spike2
            if (i + 1) % RESCALE == 0:
                x = x / 2.0
        x = layer_norm(x, w["ln_out.weight"], w["ln_out.bias"])
        if logits:
            return w["head.weight"] @ x
        return None


# ---------------------------------------------------------------- tokenizer
def load_tokenizer(outdir):
    from tokenizers import Tokenizer
    tok_path = os.path.join(outdir, "20B_tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "target/spikegpt_ref/20B_tokenizer.json"
    return Tokenizer.from_file(tok_path)


def encode_text(tok, text):
    return tok.encode(text).ids


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["convert", "ref", "gen"])
    ap.add_argument("path1", nargs="?")
    ap.add_argument("path2", nargs="?")
    ap.add_argument("--model", default="target/weights/SpikeGPT-216M.pth")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "convert":
        # Keep the documented positional form: convert <model.pth> <outdir>.
        model_path = args.path1 or args.model
        outdir = args.path2 or "target/weights"
        w = load_preprocessed(model_path)
        dump_for_rust(w, outdir)
        return

    outdir = args.path1 or "target/weights"
    w = load_preprocessed(args.model)
    os.makedirs(outdir, exist_ok=True)

    if args.tokens:
        toks = [int(t) for t in args.tokens.split()]
    else:
        tok = load_tokenizer(outdir)
        toks = encode_text(tok, args.prompt)
        print(f"prompt: {len(toks)} tokens")

    if not toks:
        raise ValueError("prompt/tokens must contain at least one token")

    torch.set_num_threads(1)          # deterministic serial reference
    runner = Runner(w)
    runner.reset()

    if args.mode == "ref":
        logits_all = []
        for t in toks:
            lg = runner.step(t, logits=True)
            logits_all.append(lg.numpy().astype(np.float32))
        logits = np.stack(logits_all)
        logits.tofile(os.path.join(outdir, "logits_ref.bin"))
        np.array(toks, dtype=np.int32).tofile(os.path.join(outdir, "tokens.txt"))
        tops = logits.argmax(-1)
        print(f"dumped logits_ref.bin {logits.shape} + tokens.txt")
        print("top-1 token ids:", tops[:20].tolist())
        return

    # gen: sample like utils.TOKENIZER.sample_logits (temperature + top-p)
    rng = np.random.RandomState(args.seed)
    out_toks = list(toks)
    # The prompt is consumed once. The logits returned for its final token
    # are the distribution used for the first generated token; do not feed
    # the final prompt token through the recurrent state a second time.
    lg = None
    for token in toks:
        lg = runner.step(token, logits=True)
    for _ in range(args.length):
        # Convert explicitly to float64: RandomState.choice requires a
        # probability sum within a tight tolerance, while float32 softmax
        # normalization can miss 1.0 by a few ulps.
        logits = lg.detach().cpu().numpy().astype(np.float64, copy=True)
        logits[0] = -999999999.0  # disable <|endoftext|>
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        cutoff = p[order][int(np.argmax(cum > 0.7))]
        p[p < cutoff] = 0.0
        p = np.power(p, 1.0 / 1.5)
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("SpikeGPT sampling produced no finite probability mass")
        p /= total
        p[-1] += 1.0 - p.sum()  # make the sum exactly 1 after rounding
        nxt = int(rng.choice(VOCAB, p=p))
        out_toks.append(nxt)
        lg = runner.step(nxt, logits=True)
    tok = load_tokenizer(outdir)
    print("GENERATED:")
    print(tok.decode(out_toks))


if __name__ == "__main__":
    main()
