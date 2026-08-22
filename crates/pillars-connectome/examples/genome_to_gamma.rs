//! # From genome to gamma
//!
//! The flagship demo of *both* engines working end-to-end:
//!
//! ```text
//!   prometheus-grow (development)            pillars (simulation)
//! ──────────────────────────────────────────────────────────────
//!   cells secrete a morphogen "A"      ──►   ConnectomeData (CSR)
//!   growth cones climb its gradient    ──►   BrainRunner, SIMD LIF-α
//!   axons meet soma → synapses               (Shiu et al. 2024 model)
//!   └─► a locally-clustered E/I graph        └─► coherent dynamics?
//! ```
//!
//! **The question:** do the *development rules alone* (diffusion + gradients +
//! contact wiring — no hand-written weights, no geometry table) build a network
//! whose *dynamics* are non-trivial? Or is any random graph with the same
//! degrees just as good?
//!
//! **The control:** take the grown connectome and shuffle the synapse targets
//! (degree sequence preserved, geometry destroyed). Same neurons, same number
//! of synapses, same E/I composition — only the developmental order is gone.
//!
//! **The test:** spike synchrony as a function of anatomical distance. A
//! development-built circuit should make nearby neurons co-fire and distant
//! ones not — a spatial correlation gradient. A geometry-free graph has
//! nothing to say about space: its correlation should be flat.
//!
//! ```bash
//! cargo run -p pillars-connectome --example genome_to_gamma --release
//! ```

use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};
use prometheus_grow::{Bounds, CellType, ChemoAttraction, Grid, Secretor, Substance, Vec3, World};
use std::time::Instant;

// ─── Experiment knobs ───────────────────────────────────────────────────────
const N_E: usize = 160; // excitatory neurons
const N_I: usize = 40; // inhibitory neurons
const N: usize = N_E + N_I;
const CLUSTER_RADIUS: f64 = 65.0; // cells placed in a ball of this radius
const SYNAPSE_DIST: f64 = 13.0; // contact radius for synapse formation
const GROW_STEPS: usize = 300;

const DT_MS: f32 = 0.5; // sim timestep
const SIM_STEPS: usize = 50_000; // 25 s of simulated time
const DRIVE: f32 = 0.2; // tonic input added to v each step (mV/step)
const W_E_GAIN: f32 = 0.15; // scale excitatory weights
const W_I_GAIN: f32 = 0.8; // scale inhibitory weights (E/I balance: I = 5× E)
const SEED: u64 = 0xC0FF_EE;

// ─── Tiny deterministic RNG (no extra deps) ─────────────────────────────────
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── 1. Development: grow the microcircuit ─────────────────────────────────
///
/// Returns (edges, nt_types, soma_positions, growth_wall_seconds).
/// `nt_types[i]` = +1 for excitatory, -1 for inhibitory.
fn grow_circuit() -> (Vec<(usize, usize, f32)>, Vec<f32>, Vec<Vec3>, f64) {
    let grid = Grid::new(48, 48, 48, 5.0, Vec3::new(-120.0, -120.0, -120.0));
    let bounds = Bounds {
        min: Vec3::new(-120.0, -120.0, -120.0),
        max: Vec3::new(120.0, 120.0, 120.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 7);
    world.synapse_dist = SYNAPSE_DIST;

    world.add_substance(Substance::new("A", 0.2, 0.001, &world.grid, vec![]));

    // Place an E/I population in a ball (uniform in volume, deterministic).
    let mut rng = XorShift(SEED);
    let mut positions: Vec<Vec3> = Vec::with_capacity(N);
    while positions.len() < N {
        // Uniform in the ball: direction × r^(1/3).
        let (u, v) = (rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 * std::f64::consts::PI);
        let c = u.cos();
        let r = CLUSTER_RADIUS * rng.unit().cbrt();
        let xy = (1.0 - u * u).max(0.0).sqrt() * r;
        positions.push(Vec3::new(xy * v.cos(), xy * v.sin(), c * r));
    }

    for (i, p) in positions.iter().enumerate() {
        let ty = if i < N_E {
            CellType::Excitatory
        } else {
            CellType::Inhibitory
        };
        let c = world.add_cell(*p, ty, 4.0);
        // Every cell secretes the shared attractant — the soma is a morphogen
        // source, exactly CX3D's self-construction idea.
        world.add_soma_module(c, Box::new(Secretor::new("A", 0.06)));
        // E cells grow two axons, I cells one: richer recurrent excitation,
        // tighter inhibition — the classic gamma recipe.
        let axons = if i < N_E { 2 } else { 1 };
        for _ in 0..axons {
            let n = world.new_neurite(c, true, 0.5);
            let mut chemo = ChemoAttraction::new("A");
            chemo.noise_weight = 0.12;
            chemo.gradient_weight = 0.35;
            chemo.stop_concentration = 0.5; // 200 sources → richer field
            chemo.max_len_um = Some(30.0); // short-range axons → local connectivity
            world.add_neurite_module(c, n, Box::new(chemo));
        }
    }

    let t0 = Instant::now();
    for _ in 0..GROW_STEPS {
        world.step();
    }
    let grow_s = t0.elapsed().as_secs_f64();

    let g = world.to_graph();
    let nt_types: Vec<f32> = g
        .nodes
        .iter()
        .map(|nd| match nd.cell_type {
            CellType::Inhibitory => -1.0,
            _ => 1.0,
        })
        .collect();
    let edges: Vec<(usize, usize, f32)> = g
        .edges
        .iter()
        .map(|e| (e.pre, e.post, e.weight as f32))
        .collect();
    (edges, nt_types, positions, grow_s)
}

// ─── 2. Bridge: edges → CSR → BrainRunner ──────────────────────────────────
fn build_runner(edges: &[(usize, usize, f32)], nt: &[f32], n: usize) -> BrainRunner {
    let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n];
    for &(pre, post, w) in edges {
        let gain = if nt[pre] > 0.0 { W_E_GAIN } else { W_I_GAIN };
        per[pre].push((post, w * gain));
    }
    let mut row_ptr = vec![0i32; n + 1];
    let mut ci = Vec::with_capacity(edges.len());
    let mut vals = Vec::with_capacity(edges.len());
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = ci.len() as i32;
        for &(post, w) in list {
            ci.push(post as i32);
            vals.push(w);
        }
    }
    row_ptr[n] = ci.len() as i32;
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt.to_vec(), vec![]);
    BrainRunner::new(conn, NeuronConfig::LifAlpha {
        dt_ms: DT_MS,
        tau_ref: 2.2,
        tau_m: 20.0,
        v_rest: -52.0,
        v_thresh: -45.0,
        v_reset: -52.0,
        tau_syn: 3.0, // faster E/I loop → gamma
        t_delay: 1.8,
    })
    .expect("BrainRunner init")
}

/// Null model: shuffle synapse *targets* (degree sequence preserved, geometry
/// destroyed). For each source, draw `d_i` distinct targets uniformly from the
/// pool — the configuration-model null. No unbounded retries: a per-source
/// Fisher–Yates over the valid pool is guaranteed to terminate.
fn shuffle_targets(edges: &[(usize, usize, f32)], n: usize, seed: u64) -> Vec<(usize, usize, f32)> {
    let mut rng = XorShift(seed);
    let mut out: Vec<(usize, usize, f32)> = Vec::with_capacity(edges.len());
    let mut cursor = 0usize;
    while cursor < edges.len() {
        let src = edges[cursor].0;
        let mut end = cursor;
        while end < edges.len() && edges[end].0 == src {
            end += 1;
        }
        let d = end - cursor;
        // Valid pool = all neurons except the source itself.
        let mut pool: Vec<usize> = (0..n).filter(|&x| x != src).collect();
        for i in (1..pool.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            pool.swap(i, j);
        }
        for k in 0..d {
            out.push((src, pool[k], edges[cursor + k].2));
        }
        cursor = end;
    }
    out
}

// ─── 3. Analysis helpers ───────────────────────────────────────────────────
/// Undirected clustering coefficient (fraction of neighbour pairs that are
/// themselves connected) — the structural signature of spatial wiring.
fn clustering_coeff(edges: &[(usize, usize, f32)], n: usize) -> f64 {
    let mut adj: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
    for &(a, b, _) in edges {
        adj[a].insert(b);
        adj[b].insert(a);
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        let neigh: Vec<usize> = adj[i].iter().copied().collect();
        let k = neigh.len();
        if k < 2 {
            continue;
        }
        let mut links = 0usize;
        for a in 0..k {
            for b in (a + 1)..k {
                if adj[neigh[a]].contains(&neigh[b]) {
                    links += 1;
                }
            }
        }
        total += 2.0 * links as f64 / (k * (k - 1)) as f64;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Population firing rate, binned every `bin_ms`.
fn population_rate(spikes: &[i32], n: usize, steps: usize, bin_ms: f32) -> Vec<f32> {
    let bin = (bin_ms / DT_MS).round().max(1.0) as usize;
    let nb = steps.div_ceil(bin);
    let mut rate = vec![0.0f32; nb];
    for t in 0..steps {
        let mut s = 0i32;
        for i in 0..n {
            s += spikes[t * n + i];
        }
        rate[t / bin] += s as f32;
    }
    let per_sec = 1000.0 / bin_ms;
    for r in rate.iter_mut() {
        *r *= per_sec / n as f32; // Hz (population average)
    }
    rate
}

/// Radix-2 iterative FFT in place (real input, complex output).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..(len / 2) {
                let u = i + k;
                let v = u + len / 2;
                let (tr, ti) = (cr * re[v] - ci * im[v], cr * im[v] + ci * re[v]);
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] += tr;
                im[u] += ti;
                let (ncr, nci) = (cr * wr - ci * wi, cr * wi + ci * wr);
                cr = ncr;
                ci = nci;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Power spectrum of the population rate: (freqs, powers).
fn spectrum(rate: &[f32], bin_ms: f32) -> (Vec<f64>, Vec<f64>) {
    let n = rate.len().next_power_of_two();
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for (i, r) in rate.iter().enumerate() {
        re[i] = *r as f64;
    }
    fft(&mut re, &mut im);
    let fs = 1000.0 / bin_ms as f64; // sample rate
    let mut freqs = Vec::with_capacity(n / 2);
    let mut powers = Vec::with_capacity(n / 2);
    for k in 0..(n / 2) {
        freqs.push(k as f64 * fs / n as f64);
        powers.push(re[k] * re[k] + im[k] * im[k]);
    }
    (freqs, powers)
}

/// Dominant frequency in [lo, hi] Hz and its peak-to-band-mean ratio
/// (spectral concentration — 1.0 = flat noise, ≫1 = oscillation).
fn dominant_freq(rate: &[f32], bin_ms: f32, lo: f64, hi: f64) -> (f64, f64) {
    let (freqs, powers) = spectrum(rate, bin_ms);
    let mut best_f = 0.0f64;
    let mut best_p = 0.0f64;
    let mut band_sum = 0.0f64;
    let mut band_n = 0usize;
    for (f, p) in freqs.iter().zip(powers.iter()) {
        if *f >= lo && *f <= hi {
            band_sum += p;
            band_n += 1;
            if *p > best_p {
                best_p = *p;
                best_f = *f;
            }
        }
    }
    if band_n == 0 || best_p == 0.0 {
        return (0.0, 1.0);
    }
    (best_f, best_p / (band_sum / band_n as f64))
}

/// Mean pairwise spike-correlation as a function of soma distance.
///
/// Bins each neuron's spike train at `bin_ms`, then for every pair computes
/// the Pearson correlation of the binned trains and the Euclidean soma
/// distance. Returns (bucket_mid, mean_corr) for `n_buckets` equal-width
/// distance buckets. Development-built circuits co-fire locally (corr decays
/// with distance); a geometry-free null model does not.
fn spatial_corr(
    spikes: &[i32],
    n: usize,
    steps: usize,
    bin_ms: f32,
    positions: &[Vec3],
    n_buckets: usize,
) -> Vec<(f64, f64)> {
    let bin = (bin_ms / DT_MS).round().max(1.0) as usize;
    let nb = steps.div_ceil(bin);
    // Binned trains (n × nb).
    let mut trains = vec![0.0f64; n * nb];
    for t in 0..steps {
        for i in 0..n {
            trains[i * nb + t / bin] += spikes[t * n + i] as f64;
        }
    }
    // Per-neuron mean.
    let mut mean = vec![0.0f64; n];
    for i in 0..n {
        mean[i] = trains[i * nb..(i + 1) * nb].iter().sum::<f64>() / nb as f64;
    }
    // Max distance → bucket width.
    let mut max_d = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = positions[i].dist(positions[j]);
            if d > max_d {
                max_d = d;
            }
        }
    }
    let bw = (max_d / n_buckets as f64).max(1e-9);
    let mut sum = vec![0.0f64; n_buckets];
    let mut cnt = vec![0usize; n_buckets];
    for i in 0..n {
        let mut vi = 0.0f64;
        for k in 0..nb {
            let d = trains[i * nb + k] - mean[i];
            vi += d * d;
        }
        if vi < 1e-12 {
            continue;
        }
        for j in (i + 1)..n {
            let mut vj = 0.0f64;
            for k in 0..nb {
                let d = trains[j * nb + k] - mean[j];
                vj += d * d;
            }
            if vj < 1e-12 {
                continue;
            }
            let mut cov = 0.0f64;
            for k in 0..nb {
                cov += (trains[i * nb + k] - mean[i]) * (trains[j * nb + k] - mean[j]);
            }
            let c = cov / (vi * vj).sqrt();
            let d = positions[i].dist(positions[j]);
            let b = ((d / bw) as usize).min(n_buckets - 1);
            sum[b] += c;
            cnt[b] += 1;
        }
    }
    (0..n_buckets)
        .map(|b| {
            let mid = (b as f64 + 0.5) * bw;
            let m = if cnt[b] > 0 { sum[b] / cnt[b] as f64 } else { 0.0 };
            (mid, m)
        })
        .collect()
}


/// Oscillation strength: peak of the normalized autocorrelation of the
/// smoothed rate (0 = noise, ~1 = perfectly periodic).
fn osc_strength(rate: &[f32], bin_ms: f32) -> f64 {
    let win: Vec<f32> = (0..(20.0 / bin_ms) as usize).map(|_| 1.0).collect();
    let n = rate.len();
    let hw = win.len() / 2;
    let mut s = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0;
        for (k, &w) in win.iter().enumerate() {
            let j = i as isize + k as isize - hw as isize;
            if j >= 0 && j < n as isize {
                acc += rate[j as usize] * w;
            }
        }
        s[i] = acc;
    }
    let mean = s.iter().sum::<f32>() / n as f32;
    let mut var = 0.0f64;
    for x in &s {
        var += (*x as f64 - mean as f64).powi(2);
    }
    if var < 1e-9 {
        return 0.0;
    }
    let max_lag = ((200.0 / bin_ms) as usize).min(n / 4);
    let min_lag = ((10.0 / bin_ms) as usize).max(1);
    let mut best = 0.0f64;
    for lag in min_lag..max_lag {
        let mut acc = 0.0f64;
        for i in 0..(n - lag) {
            acc += (s[i] as f64 - mean as f64) * (s[i + lag] as f64 - mean as f64);
        }
        let c = acc / var;
        if c > best {
            best = c;
        }
    }
    best
}
// ─── 4. Main ───────────────────────────────────────────────────────────────
fn main() {
    println!("═══ FROM GENOME TO GAMMA ═══════════════════════════════════");
    println!("development rules → structure → function, end to end\n");

    // ── Grow ──────────────────────────────────────────────────────────────
    let (edges, nt, positions, grow_s) = grow_circuit();
    let n_syn = edges.len();
    let n_e_eff = nt.iter().filter(|v| **v > 0.0).count();
    let n_i_eff = N - n_e_eff;
    println!("[1] development  (prometheus-grow, {GROW_STEPS} steps)");
    println!(
        "    {N} neurons ({n_e_eff} E / {n_i_eff} I) → {n_syn} synapses in {grow_s:.2}s"
    );

    let cc_grown = clustering_coeff(&edges, N);
    let mut pre_e = 0usize;
    let mut pre_i = 0usize;
    for &(pre, _, _) in &edges {
        if nt[pre] > 0.0 {
            pre_e += 1;
        } else {
            pre_i += 1;
        }
    }
    println!(
        "    clustering C = {cc_grown:.3} | E-driven {:.0}% | I-driven {:.0}%",
        pre_e as f64 * 100.0 / n_syn as f64,
        pre_i as f64 * 100.0 / n_syn as f64
    );

    // ── Control: same degrees, geometry destroyed ─────────────────────────
    let shuffled = shuffle_targets(&edges, N, SEED ^ 0xDEAD_BEEF);
    let cc_shuf = clustering_coeff(&shuffled, N);
    println!(
        "[2] control      shuffled targets: C = {cc_shuf:.3} (same degree sequence)"
    );

    // ── Simulate both ─────────────────────────────────────────────────────
    let mut grown = build_runner(&edges, &nt, N);
    let mut shuf = build_runner(&shuffled, &nt, N);
    let ext = vec![DRIVE; N];

    let t0 = Instant::now();
    let sg = grown.run(SIM_STEPS, &ext).to_vec();
    let sim_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let ss = shuf.run(SIM_STEPS, &ext).to_vec();
    let sim_s_shuf = t0.elapsed().as_secs_f64();

    let rates_g = grown.firing_rates(DT_MS);
    let rates_s = shuf.firing_rates(DT_MS);
    let mean_g = rates_g.iter().sum::<f32>() / N as f32;
    let mean_s = rates_s.iter().sum::<f32>() / N as f32;
    let rate_e: f32 = rates_g[..N_E].iter().sum::<f32>() / N_E as f32;
    let rate_i: f32 = rates_g[N_E..].iter().sum::<f32>() / N_I as f32;

    // ── Dynamics analysis ─────────────────────────────────────────────────
    let bin_ms = 4.0;
    let pg = population_rate(&sg, N, SIM_STEPS, bin_ms);
    let ps = population_rate(&ss, N, SIM_STEPS, bin_ms);
    let (fg, cg) = dominant_freq(&pg, bin_ms, 20.0, 120.0);
    let (fs, cs) = dominant_freq(&ps, bin_ms, 20.0, 120.0);
    let og = osc_strength(&pg, bin_ms);
    let os = osc_strength(&ps, bin_ms);

    let cell_steps = N as f64 * SIM_STEPS as f64;
    println!("\n[3] simulation   (pillars, SIMD LIF-α, {SIM_STEPS} steps × {N} neurons)");
    println!(
        "    grown: {:.3}s | shuffled: {:.3}s | {:.0} M cell-steps/s (grown)",
        sim_s,
        sim_s_shuf,
        cell_steps / sim_s / 1e6
    );

    println!("\n[4] population dynamics (binned {bin_ms:.0} ms)");
    println!("    ┌───────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("    │           │  rate Hz │ dom. freq│ peak/bg  │  osc     │");
    println!("    ├───────────┼──────────┼──────────┼──────────┼──────────┤");
    println!(
        "    │ grown     │ {:>8.1} │ {:>8.1} │ {:>8.2} │ {:>8.3} │",
        mean_g, fg, cg, og
    );
    println!(
        "    │ shuffled  │ {:>8.1} │ {:>8.1} │ {:>8.2} │ {:>8.3} │",
        mean_s, fs, cs, os
    );
    println!("    └───────────┴──────────┴──────────┴──────────┴──────────┘");
    println!(
        "    E rate {rate_e:.1} Hz, I rate {rate_i:.1} Hz (E/I balance {:.2})",
        rate_e / rate_i.max(0.1)
    );

    // ── Spatial synchrony: the geometry test ──────────────────────────────
    let scg = spatial_corr(&sg, N, SIM_STEPS, 50.0, &positions, 5);
    let scs = spatial_corr(&ss, N, SIM_STEPS, 50.0, &positions, 5);
    println!("\n[5] spatial synchrony — spike corr. vs. soma distance (50 ms bins)");
    println!("    ┌───────────┬──────────┬──────────┬──────────┐");
    println!("    │ distance  │  grown   │ shuffled │  ratio   │");
    println!("    ├───────────┼──────────┼──────────┼──────────┤");
    for b in 0..scg.len() {
        let (d, cgv) = scg[b];
        let (_, csv) = scs[b];
        println!(
            "    │ {:>7.0} µm │ {:>8.3} │ {:>8.3} │ {:>8.2} │",
            d,
            cgv,
            csv,
            cgv / csv.max(1e-6)
        );
    }
    println!("    └───────────┴──────────┴──────────┴──────────┘");

    let (d0, c0) = scg[0];
    let (d1, c1) = *scg.last().unwrap();
    let (_, s0) = scs[0];
    let (_, s1) = *scs.last().unwrap();
    let decay_g = c0 - c1;
    let decay_s = s0 - s1;
    let gamma = (20.0..=80.0).contains(&fg);
    println!("\n[6] verdict");
    if decay_g > decay_s + 0.03 && c0 > s0 {
        println!(
            "    ✅ Development rules built spatially coherent assemblies: neurons within\n    \
             {d0:.0} µm co-fire at r={c0:.2} (vs r={s0:.2} shuffled), while {d1:.0} µm apart\n    \
             the correlation drops to r={c1:.2} — a gradient of {decay_g:.2}. The shuffled\n    \
             control has no spatial structure (gradient {decay_s:.2})."
        );
        if gamma {
            println!(
                "    Bonus: the population oscillates — a sharp beta/gamma rhythm: {fg:.0} Hz, with a spectral
                 peak {cg:.1}× above background)."
            );
        }
        println!(
            "    Geometry matters: destroy it (same degrees!) and co-firing collapses."
        );
    } else {
        println!(
            "    ⚠️  weak spatial contrast (grown {c0:.2}→{c1:.2}, shuffled {s0:.2}→{s1:.2}). \
             Tune drive/weights."
        );
    }

    // ── Export population rate for the viewer ─────────────────────────────
    if let Ok(mut f) = std::fs::File::create("target/genome_to_gamma_rates.csv") {
        use std::io::Write;
        let _ = writeln!(f, "t_ms,grown_hz,shuffled_hz");
        for (t, (a, b)) in pg.iter().zip(ps.iter()).enumerate() {
            let _ = writeln!(f, "{},{:.2},{:.2}", t as f32 * bin_ms, a, b);
        }
        println!("\n    rates → target/genome_to_gamma_rates.csv (for the viewer)");
    }
}
