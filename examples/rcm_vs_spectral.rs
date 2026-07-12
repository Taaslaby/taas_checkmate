//! RCM vs Spectral ordering bandwidth comparison on FlyWire-scale connectome.
//!
//! Usage: cargo run --release --example rcm_vs_spectral --features connectome

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     RCM vs Spectral Ordering — FlyWire-scale Connectome    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    for &(n, n_syn, label) in &[
        (10_000, 10, "10K neurons"),
        (50_000, 10, "50K neurons"),
        (100_000, 10, "100K neurons (FlyWire-scale)"),
    ] {
        println!("\n═══ {label} ═══");

        // Generate connectome (inline RNG to avoid SmallRng feature)
        let conn = generate_connectome(n, n_syn);
        let avg_syn = conn.values.len() / n;
        println!("  Synapses/neuron: ~{avg_syn}");

        // Bandwidth before any ordering
        let bw_before = taas_connectome::bandwidth(&conn.row_ptr, &conn.col_indices);
        let mbw_before = taas_connectome::mean_bandwidth(&conn.row_ptr, &conn.col_indices);
        println!("  ── BEFORE ordering ──");
        println!("    Bandwidth:     {bw_before:>10}");
        println!("    Mean BW:       {mbw_before:>10.2}");

        // ─── RCM ──────────────────────────────────────────────────────
        println!("  ── RCM ──");
        let t0 = Instant::now();
        let adj = taas_connectome::build_adjacency(&conn);
        let rcm_perm = taas_connectome::reverse_cuthill_mckee(&adj);
        let (rp_rcm, ci_rcm, _, _) = taas_connectome::apply_permutation_to_csr(
            &conn.row_ptr, &conn.col_indices, &conn.values,
            &conn.nt_types, &rcm_perm,
        );
        let rcm_time = t0.elapsed();

        let bw_rcm = taas_connectome::bandwidth(&rp_rcm, &ci_rcm);
        let mbw_rcm = taas_connectome::mean_bandwidth(&rp_rcm, &ci_rcm);
        println!("    Time:          {:.3}s", rcm_time.as_secs_f64());
        println!("    Bandwidth:     {bw_rcm:>10}  ({:.0}× reduction)",
            bw_before as f64 / bw_rcm.max(1) as f64);
        println!("    Mean BW:       {mbw_rcm:>10.2}  ({:.0}× reduction)",
            mbw_before / mbw_rcm.max(1e-10));

        // ─── Spectral ordering ────────────────────────────────────────
        println!("  ── Spectral (Fiedler) ──");
        let t0 = Instant::now();
        let spec_perm = taas_connectome::spectral_ordering(&adj, 50, 1e-3);
        let (rp_spec, ci_spec, _, _) = taas_connectome::apply_permutation_to_csr(
            &conn.row_ptr, &conn.col_indices, &conn.values,
            &conn.nt_types, &spec_perm,
        );
        let spec_time = t0.elapsed();

        let bw_spec = taas_connectome::bandwidth(&rp_spec, &ci_spec);
        let mbw_spec = taas_connectome::mean_bandwidth(&rp_spec, &ci_spec);
        println!("    Time:          {:.3}s", spec_time.as_secs_f64());
        println!("    Bandwidth:     {bw_spec:>10}  ({:.0}× reduction)",
            bw_before as f64 / bw_spec.max(1) as f64);
        println!("    Mean BW:       {mbw_spec:>10.2}  ({:.0}× reduction)",
            mbw_before / mbw_spec.max(1e-10));

        // ─── Comparison ───────────────────────────────────────────────
        println!("  ── RCM vs Spectral ──");
        let bw_ratio = if bw_rcm > 0 { bw_spec as f64 / bw_rcm as f64 } else { 0.0 };
        let mbw_ratio = if mbw_rcm > 0.0 { mbw_spec / mbw_rcm } else { 0.0 };

        if bw_ratio < 1.0 {
            println!("    Bandwidth:     Spectral is {:.1}× BETTER than RCM", 1.0 / bw_ratio);
        } else {
            println!("    Bandwidth:     RCM is {:.1}× BETTER than Spectral", bw_ratio);
        }
        if mbw_ratio < 1.0 {
            println!("    Mean BW:       Spectral is {:.1}× BETTER than RCM", 1.0 / mbw_ratio);
        } else {
            println!("    Mean BW:       RCM is {:.1}× BETTER than Spectral", mbw_ratio);
        }

        if rcm_time < spec_time {
            println!("    Preprocessing: RCM is {:.1}× faster than Spectral",
                spec_time.as_secs_f64() / rcm_time.as_secs_f64());
        } else {
            println!("    Preprocessing: Spectral is {:.1}× faster than RCM",
                rcm_time.as_secs_f64() / spec_time.as_secs_f64());
        }
    }
}

// ─── Inline RNG (no SmallRng dependency) ───────────────────────────────────
fn rng_u32(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}
fn rng_f32(state: &mut u64) -> f32 {
    rng_u32(state) as f32 / 4294967296.0
}
fn rng_bool(state: &mut u64, p: f64) -> bool {
    rng_f32(state) < p as f32
}
fn rng_range(state: &mut u64, lo: usize, hi: usize) -> usize {
    lo + (rng_f32(state) * (hi - lo) as f32) as usize
}

fn generate_connectome(n: usize, n_remote_syn: usize) -> taas_connectome::ConnectomeData {
    let mut state: u64 = 42;
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut syn_count = 0i32;

    for i in 0..n {
        // Ring: i → i+1
        col_indices.push(((i + 1) % n) as i32);
        values.push(0.5);
        syn_count += 1;

        for _ in 0..n_remote_syn {
            let target = if rng_bool(&mut state, 0.7) {
                let offset = n / 4 + rng_range(&mut state, 0, n / 2);
                (i + offset) % n
            } else {
                let offset = 1 + rng_range(&mut state, 0, 32);
                (i + offset) % n
            };
            col_indices.push(target as i32);
            values.push(0.1 + rng_f32(&mut state) * 1.9);
            syn_count += 1;
        }
        row_ptr[i + 1] = syn_count;
    }

    taas_connectome::ConnectomeData {
        n_neurons: n,
        n_synapses: values.len(),
        row_ptr,
        col_indices,
        values,
        nt_types: vec![1.0f32; n],
        groups: vec![],
    }
}
