//! SpikeGPT-216M inference on the Taas engine — exact Rust port of the
//! reference runner `ridgerchu/SpikeGPT` `src/model_run.py` (RWKV_RNN form,
//! fp32, B=1). Parity-verified against the torch CPU reference.
//!
//! Usage:
//!   cargo run --example spikegpt_infer --release -- --parity --dir target/weights
//!   cargo run --example spikegpt_infer --release -- --temporal --dir target/weights
//!
//! Temporal mode accepts reproducible token overrides:
//!   --anchor ID --second ID --delay-token ID --query ID
//! It measures discrete recurrent state retention, not subjective time.
//!
//! Reads: `target/weights/weights.bin` + `manifest.json` (produced by
//! `resources/examples/spikegpt_ref.py convert`), `tokens.txt` +
//! `logits_ref.bin` (produced by `spikegpt_ref.py ref`).

use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

const N_EMBD: usize = 768;
const VOCAB: usize = 50277;
const RESCALE: usize = 6;

/// Tensors loaded from weights.bin + manifest.json (f32, preprocessed).
struct Weights {
    blob: Vec<f32>,
    tensors: HashMap<String, (usize, Vec<usize>)>, // name -> (f32 offset, shape)
}

impl Weights {
    fn load(dir: &str) -> Self {
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .expect("manifest.json");
        let bytes = fs::read(format!("{dir}/weights.bin")).expect("weights.bin");
        assert_eq!(bytes.len() % 4, 0, "weights.bin must be f32-aligned size");
        let blob: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut tensors = HashMap::new();
        for (name, meta) in manifest["tensors"].as_object().unwrap() {
            let off = meta["offset"].as_u64().unwrap() as usize / 4;
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            tensors.insert(name.clone(), (off, shape));
        }
        Self { blob, tensors }
    }

    fn get(&self, name: &str) -> &[f32] {
        let (off, shape) = &self.tensors[name];
        let len: usize = shape.iter().product();
        &self.blob[*off..*off + len]
    }
}

/// out[j] = sum_t w[j*k+t] * x[t]  (w is (n x k) row-major), parallel over rows.
fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    out.par_iter_mut().enumerate().for_each(|(j, o)| {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0.0f32;
        for t in 0..k {
            acc += row[t] * x[t];
        }
        *o = acc;
    });
}

/// torch F.layer_norm(x, (C,), weight, bias, eps=1e-5) — biased variance.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// spikingjelly LIFNode(tau=2, decay_input=True, v_reset=0.0) — one step.
/// v += (x - v)/2; spike = v >= 1; hard reset v = 0 on spike.
fn lif_step(v: &mut [f32], x: &[f32], spike: &mut [f32]) {
    for i in 0..x.len() {
        let vv = v[i] + (x[i] - v[i]) * 0.5;
        let s = if vv >= 1.0 { 1.0 } else { 0.0 };
        v[i] = if s > 0.0 { 0.0 } else { vv };
        spike[i] = s;
    }
}

/// Named scratch buffers (disjoint fields => clean borrows).
struct Bufs {
    x: Vec<f32>,   // residual stream
    xin: Vec<f32>, // ln1 output / SA time-mix input
    xk: Vec<f32>,
    xv: Vec<f32>,
    xr: Vec<f32>,
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ww: Vec<f32>,
    p: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    wkv: Vec<f32>,
    rw1: Vec<f32>,
    att_out: Vec<f32>,
    spike1: Vec<f32>,
    xin2: Vec<f32>, // ln2 output / FF time-mix input
    xk2: Vec<f32>,
    xr2: Vec<f32>,
    r2: Vec<f32>,
    kv: Vec<f32>,
    ffn_out: Vec<f32>,
    spike2: Vec<f32>,
    ln_tmp: Vec<f32>,
    kk: Vec<f32>, // ffn hidden (4*768)
}

impl Bufs {
    fn new() -> Self {
        let v = || vec![0.0f32; N_EMBD];
        Self {
            x: v(),
            xin: v(),
            xk: v(),
            xv: v(),
            xr: v(),
            r: v(),
            k: v(),
            v: v(),
            ww: v(),
            p: v(),
            e1: v(),
            e2: v(),
            a: v(),
            b: v(),
            wkv: v(),
            rw1: v(),
            att_out: v(),
            spike1: v(),
            xin2: v(),
            xk2: v(),
            xr2: v(),
            r2: v(),
            kv: v(),
            ffn_out: v(),
            spike2: v(),
            ln_tmp: v(),
            kk: vec![0.0; 4 * N_EMBD],
        }
    }
}

/// Summary of the two spiking branches emitted by one token step.
#[derive(Clone, Copy, Debug)]
struct StepTrace {
    attention_spikes: usize,
    ffn_spikes: usize,
}

/// Stateful per-token forward (B=1), replica of RWKV_RNN.forward.
struct Runner<'a> {
    w: &'a Weights,
    state: Vec<f32>, // n_layer*5 x n_embd: +0 ffn_xx, +1 att_xx, +2 aa, +3 bb, +4 pp
    lif1: Vec<f32>,  // n_layer x n_embd
    lif2: Vec<f32>,
    bufs: Bufs,
}

impl<'a> Runner<'a> {
    fn new(w: &'a Weights) -> Self {
        let mut state = vec![0.0f32; 18 * 5 * N_EMBD];
        for i in 0..18 {
            let pp = (5 * i + 4) * N_EMBD;
            for value in &mut state[pp..pp + N_EMBD] {
                *value = -1e30; // pp starts at -1e30 for every channel
            }
        }
        Self {
            w,
            state,
            lif1: vec![0.0; 18 * N_EMBD],
            lif2: vec![0.0; 18 * N_EMBD],
            bufs: Bufs::new(),
        }
    }

    fn step(&mut self, tok: usize, logits: &mut [f32]) {
        let _ = self.step_with_trace(tok, logits);
    }

    fn step_with_trace(&mut self, tok: usize, logits: &mut [f32]) -> StepTrace {
        let w = self.w;
        let st = &mut self.state;
        let lif1 = &mut self.lif1;
        let lif2 = &mut self.lif2;
        let b = &mut self.bufs;
        let mut attention_spikes = 0usize;
        let mut ffn_spikes = 0usize;

        let emb = w.get("emb.weight");
        b.x.copy_from_slice(&emb[tok * N_EMBD..(tok + 1) * N_EMBD]);

        for i in 0..18 {
            if i == 0 {
                layer_norm(
                    &b.x,
                    w.get("blocks.0.ln0.weight"),
                    w.get("blocks.0.ln0.bias"),
                    &mut b.ln_tmp,
                );
                b.x.copy_from_slice(&b.ln_tmp);
            }
            // ================= SA =================
            layer_norm(
                &b.x,
                w.get(&format!("blocks.{i}.ln1.weight")),
                w.get(&format!("blocks.{i}.ln1.bias")),
                &mut b.xin,
            );
            let att_xx = st[(5 * i + 1) * N_EMBD..(5 * i + 2) * N_EMBD].to_vec();
            let tm_k = w.get(&format!("blocks.{i}.att.time_mix_k"));
            let tm_v = w.get(&format!("blocks.{i}.att.time_mix_v"));
            let tm_r = w.get(&format!("blocks.{i}.att.time_mix_r"));
            for c in 0..N_EMBD {
                b.xk[c] = b.xin[c] * tm_k[c] + att_xx[c] * (1.0 - tm_k[c]);
                b.xv[c] = b.xin[c] * tm_v[c] + att_xx[c] * (1.0 - tm_v[c]);
                b.xr[c] = b.xin[c] * tm_r[c] + att_xx[c] * (1.0 - tm_r[c]);
            }
            st[(5 * i + 1) * N_EMBD..(5 * i + 2) * N_EMBD].copy_from_slice(&b.xin);

            matvec(
                w.get(&format!("blocks.{i}.att.receptance.weight")),
                &b.xr,
                &mut b.r,
            );
            matvec(
                w.get(&format!("blocks.{i}.att.key.weight")),
                &b.xk,
                &mut b.k,
            );
            matvec(
                w.get(&format!("blocks.{i}.att.value.weight")),
                &b.xv,
                &mut b.v,
            );
            for c in 0..N_EMBD {
                b.r[c] = 1.0 / (1.0 + (-b.r[c]).exp());
            }

            let tf = w.get(&format!("blocks.{i}.att.time_first"));
            let td = w.get(&format!("blocks.{i}.att.time_decay"));
            let aa = st[(5 * i + 2) * N_EMBD..(5 * i + 3) * N_EMBD].to_vec();
            let bb = st[(5 * i + 3) * N_EMBD..(5 * i + 4) * N_EMBD].to_vec();
            let pp = st[(5 * i + 4) * N_EMBD..(5 * i + 5) * N_EMBD].to_vec();
            for c in 0..N_EMBD {
                b.ww[c] = tf[c] + b.k[c];
                b.p[c] = if pp[c] > b.ww[c] { pp[c] } else { b.ww[c] };
                b.e1[c] = (pp[c] - b.p[c]).exp();
                b.e2[c] = (b.ww[c] - b.p[c]).exp();
                b.a[c] = b.e1[c] * aa[c] + b.e2[c] * b.v[c];
                b.b[c] = b.e1[c] * bb[c] + b.e2[c];
            }
            for c in 0..N_EMBD {
                let ww2 = pp[c] + td[c];
                let p2 = if ww2 > b.k[c] { ww2 } else { b.k[c] };
                let e1b = (ww2 - p2).exp();
                let e2b = (b.k[c] - p2).exp();
                st[(5 * i + 2) * N_EMBD + c] = e1b * aa[c] + e2b * b.v[c];
                st[(5 * i + 3) * N_EMBD + c] = e1b * bb[c] + e2b;
                st[(5 * i + 4) * N_EMBD + c] = p2;
            }
            for c in 0..N_EMBD {
                b.wkv[c] = b.a[c] / b.b[c];
                b.rw1[c] = b.r[c] * b.wkv[c];
            }
            matvec(
                w.get(&format!("blocks.{i}.att.output.weight")),
                &b.rw1,
                &mut b.att_out,
            );
            lif_step(
                &mut lif1[i * N_EMBD..(i + 1) * N_EMBD],
                &b.att_out,
                &mut b.spike1,
            );
            attention_spikes += b.spike1.iter().filter(|&&s| s > 0.5).count();
            for c in 0..N_EMBD {
                b.x[c] += b.spike1[c];
            }

            // ================= FF =================
            layer_norm(
                &b.x,
                w.get(&format!("blocks.{i}.ln2.weight")),
                w.get(&format!("blocks.{i}.ln2.bias")),
                &mut b.xin2,
            );
            let ffn_xx = st[5 * i * N_EMBD..(5 * i + 1) * N_EMBD].to_vec();
            let tm_k2 = w.get(&format!("blocks.{i}.ffn.time_mix_k"));
            let tm_r2 = w.get(&format!("blocks.{i}.ffn.time_mix_r"));
            for c in 0..N_EMBD {
                b.xk2[c] = b.xin2[c] * tm_k2[c] + ffn_xx[c] * (1.0 - tm_k2[c]);
                b.xr2[c] = b.xin2[c] * tm_r2[c] + ffn_xx[c] * (1.0 - tm_r2[c]);
            }
            st[5 * i * N_EMBD..(5 * i + 1) * N_EMBD].copy_from_slice(&b.xin2);

            matvec(
                w.get(&format!("blocks.{i}.ffn.receptance.weight")),
                &b.xr2,
                &mut b.r2,
            );
            for c in 0..N_EMBD {
                b.r2[c] = 1.0 / (1.0 + (-b.r2[c]).exp());
            }
            matvec(
                w.get(&format!("blocks.{i}.ffn.key.weight")),
                &b.xk2,
                &mut b.kk,
            );
            for c in 0..(4 * N_EMBD) {
                let q = if b.kk[c] > 0.0 { b.kk[c] } else { 0.0 };
                b.kk[c] = q * q;
            }
            matvec(
                w.get(&format!("blocks.{i}.ffn.value.weight")),
                &b.kk,
                &mut b.kv,
            );
            for c in 0..N_EMBD {
                b.ffn_out[c] = b.r2[c] * b.kv[c];
            }
            lif_step(
                &mut lif2[i * N_EMBD..(i + 1) * N_EMBD],
                &b.ffn_out,
                &mut b.spike2,
            );
            ffn_spikes += b.spike2.iter().filter(|&&s| s > 0.5).count();
            for c in 0..N_EMBD {
                b.x[c] += b.spike2[c];
            }
            if (i + 1) % RESCALE == 0 {
                for c in 0..N_EMBD {
                    b.x[c] *= 0.5;
                }
            }
        }

        layer_norm(
            &b.x,
            w.get("ln_out.weight"),
            w.get("ln_out.bias"),
            &mut b.ln_tmp,
        );
        matvec(w.get("head.weight"), &b.ln_tmp, logits);
        StepTrace {
            attention_spikes,
            ffn_spikes,
        }
    }
}

fn read_i32(path: &str) -> Vec<i32> {
    let bytes = fs::read(path).expect(path);
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect(path);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let temporal = args.iter().any(|arg| arg == "--temporal");
    let parity = args.iter().any(|arg| arg == "--parity");
    if temporal && parity {
        eprintln!("error: choose exactly one of --temporal or --parity");
        std::process::exit(2);
    }
    if !temporal && !parity {
        eprintln!("usage: spikegpt_infer --parity [--dir target/weights]");
        eprintln!("       spikegpt_infer --temporal [--dir target/weights]");
        std::process::exit(2);
    }
    let dir = args
        .iter()
        .position(|a| a == "--dir")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("target/weights");

    let w = Weights::load(dir);

    let toks = read_i32(&format!("{dir}/tokens.txt"));
    assert!(
        !toks.is_empty(),
        "tokens.txt must contain at least one token"
    );
    assert!(
        toks.iter().all(|&token| (0..VOCAB as i32).contains(&token)),
        "tokens.txt contains a token outside the SpikeGPT vocabulary"
    );
    if temporal {
        let overrides = ["--anchor", "--second", "--delay-token", "--query"]
            .map(|flag| parse_usize_flag(&args, flag));
        let [anchor, second, delay_token, query] = overrides;
        let overrides = match (anchor, second, delay_token, query) {
            (Ok(anchor), Ok(second), Ok(delay_token), Ok(query)) => {
                (anchor, second, delay_token, query)
            }
            (Err(error), _, _, _)
            | (_, Err(error), _, _)
            | (_, _, Err(error), _)
            | (_, _, _, Err(error)) => {
                eprintln!("error: {error}");
                std::process::exit(2);
            }
        };
        for (name, token) in [
            ("anchor", overrides.0),
            ("second", overrides.1),
            ("delay-token", overrides.2),
            ("query", overrides.3),
        ] {
            if let Some(token) = token {
                if token >= VOCAB {
                    eprintln!("error: {name} token {token} is outside vocabulary size {VOCAB}");
                    std::process::exit(2);
                }
            }
        }
        run_temporal(
            &w,
            &toks,
            overrides.0,
            overrides.1,
            overrides.2,
            overrides.3,
        );
        return;
    }

    let mut runner = Runner::new(&w);

    let toks = toks;
    let ref_logits = read_f32(&format!("{dir}/logits_ref.bin"));
    assert_eq!(
        ref_logits.len(),
        toks.len() * VOCAB,
        "logits_ref.bin size mismatch"
    );

    let mut logits = vec![0.0f32; VOCAB];
    let mut max_abs: f32 = 0.0;
    let mut top1_hits = 0usize;
    let mut spike_counts = [[0usize; 2]; 18];

    println!("parity over {} tokens (VOCAB={VOCAB})", toks.len());
    let t0 = std::time::Instant::now();
    for (t, tok) in toks.iter().enumerate() {
        runner.step(*tok as usize, &mut logits);
        let base = t * VOCAB;
        let ref_top = (0..VOCAB)
            .max_by(|&a, &b| ref_logits[base + a].total_cmp(&ref_logits[base + b]))
            .unwrap();
        let my_top = (0..VOCAB)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap();
        let mut m = 0.0f32;
        for j in 0..VOCAB {
            let d = (logits[j] - ref_logits[base + j]).abs();
            if d > m {
                m = d;
            }
        }
        if my_top == ref_top {
            top1_hits += 1;
        }
        max_abs = max_abs.max(m);
        if t < 6 {
            println!(
                "  tok[{t}] id={tok} top1={my_top} ref_top1={ref_top} match={} maxabs={m:.4}",
                my_top == ref_top
            );
        }
        if t == toks.len() - 1 {
            for i in 0..18 {
                spike_counts[i][0] = lif_count(&runner.lif1, i);
                spike_counts[i][1] = lif_count(&runner.lif2, i);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "\nRESULT: top-1 agreement {top1_hits}/{} ({}%), max abs logit diff {max_abs:.4}",
        toks.len(),
        top1_hits * 100 / toks.len()
    );
    println!(
        "time: {dt:.1}s ({:.1} ms/token)",
        dt / toks.len() as f64 * 1e3
    );
    assert_eq!(
        top1_hits,
        toks.len(),
        "SpikeGPT parity failed: top-1 logits mismatch"
    );
    assert!(
        max_abs <= 1e-2,
        "SpikeGPT parity failed: max abs logit diff {max_abs:.6} > 0.01"
    );
    println!("last-step LIF spike counts per layer (att, ffn):");
    for i in 0..18 {
        println!(
            "  layer {i:2}: att={:4} ffn={:4}",
            spike_counts[i][0], spike_counts[i][1]
        );
    }
}

fn lif_count(lif: &[f32], i: usize) -> usize {
    lif[i * N_EMBD..(i + 1) * N_EMBD]
        .iter()
        .filter(|v| **v > 0.5)
        .count()
}

fn run_sequence(w: &Weights, tokens: &[usize]) -> (Vec<f32>, StepTrace) {
    let mut runner = Runner::new(w);
    let mut logits = vec![0.0f32; VOCAB];
    let mut trace = StepTrace {
        attention_spikes: 0,
        ffn_spikes: 0,
    };
    for &token in tokens {
        trace = runner.step_with_trace(token, &mut logits);
    }
    (logits, trace)
}

fn rms_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>();
    (sum / a.len() as f32).sqrt()
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn parse_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>, String> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{flag} requires an integer value"))?;
    value
        .parse::<usize>()
        .map(Some)
        .map_err(|_| format!("{flag} requires an integer value, got '{value}'"))
}

/// Measure discrete temporal memory in the real pretrained recurrent model.
/// This deliberately does not call the result "time perception": it measures
/// dependence on token order and on the number of recurrent update steps.
fn run_temporal(
    w: &Weights,
    source_tokens: &[i32],
    anchor_override: Option<usize>,
    second_override: Option<usize>,
    delay_override: Option<usize>,
    query_override: Option<usize>,
) {
    let ids: Vec<usize> = source_tokens.iter().map(|&id| id as usize).collect();
    let anchor = anchor_override.unwrap_or(ids[0]);
    let second = second_override.unwrap_or_else(|| ids.get(1).copied().unwrap_or(anchor));
    let delay_token = delay_override.unwrap_or_else(|| ids.get(2).copied().unwrap_or(anchor));
    let query = query_override.unwrap_or_else(|| ids.last().copied().unwrap_or(anchor));
    let (fresh, fresh_trace) = run_sequence(w, &[query]);

    println!("SpikeGPT temporal experiment (real pretrained recurrent state)");
    println!("anchor={anchor}, second={second}, delay_token={delay_token}, query={query}");
    println!("baseline: query from a fresh state");
    println!("delay control: the same delay tokens plus query, but without anchor");
    println!();

    let (ab, ab_trace) = run_sequence(w, &[anchor, second, query]);
    let (ba, ba_trace) = run_sequence(w, &[second, anchor, query]);
    println!("order sensitivity: [A,B,Q] vs [B,A,Q]");
    println!(
        "  rms_logit_delta={:.5}, max_logit_delta={:.5}, top1={} vs {}, query_spikes=(att {}, ffn {}) / (att {}, ffn {})",
        rms_delta(&ab, &ba),
        max_abs_delta(&ab, &ba),
        argmax(&ab),
        argmax(&ba),
        ab_trace.attention_spikes,
        ab_trace.ffn_spikes,
        ba_trace.attention_spikes,
        ba_trace.ffn_spikes,
    );
    println!();

    println!("state retention after a discrete delay");
    println!("  delay_steps  rms_vs_fresh  anchor_rms  anchor_max  top1  query_spikes(att/ffn)");
    for delay_steps in [0usize, 1, 2, 4, 8, 16, 32, 64] {
        let mut anchored = Vec::with_capacity(delay_steps + 2);
        anchored.push(anchor);
        anchored.extend(std::iter::repeat_n(delay_token, delay_steps));
        anchored.push(query);

        let mut control = Vec::with_capacity(delay_steps + 2);
        control.extend(std::iter::repeat_n(delay_token, delay_steps + 1));
        control.push(query);

        let (context_logits, trace) = run_sequence(w, &anchored);
        let (control_logits, _) = run_sequence(w, &control);
        println!(
            "  {:11}  {:13.5}  {:10.5}  {:10.5}  {:4}  {}/{}",
            delay_steps,
            rms_delta(&context_logits, &fresh),
            rms_delta(&context_logits, &control_logits),
            max_abs_delta(&context_logits, &control_logits),
            argmax(&context_logits),
            trace.attention_spikes,
            trace.ffn_spikes,
        );
    }
    println!();
    println!("interpretation:");
    println!("  • nonzero order delta = recurrent state preserves sequence order;");
    println!("  • anchor_rms compares against the same delay without anchor;");
    println!("  • changing anchor_rms across delays measures context retention/decay;");
    println!("  • spike-count changes = the temporal context changes LIF activity;");
    println!("  • this is discrete state retention/decay, not evidence of subjective time.");
    println!(
        "fresh query spikes: att {}, ffn {}",
        fresh_trace.attention_spikes, fresh_trace.ffn_spikes
    );
}
