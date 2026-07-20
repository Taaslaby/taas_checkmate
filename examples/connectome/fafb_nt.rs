/// Test the new .taas format with receptor_types on real FAFB data.
///
/// Loads fafb.taas (old format, no receptor_types) via `ConnectomeData::load_baked()`,
/// builds per-synapse receptor_types from nt_types, saves to new format,
/// loads back and verifies roundtrip.
///
/// This validates:
///   1. Backward compat: loading old .taas without receptor_types
///   2. build_receptor_types() on a real connectome
///   3. save_baked() with receptor_types (new format)
///   4. load_baked() roundtrip of new format
///
/// Usage:
///   cargo run --features connectome --example fafb_nt --release
///
/// Requires: fafb.taas in project root (nt_types already patched with E/I).

#[cfg(feature = "connectome")]
mod inner {
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Instant;
    use taas_connectome::ConnectomeData;

    fn data(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("examples/data");
        p.push(name);
        p
    }

    pub fn run() {
        let t0 = Instant::now();

        // ─── 1. Load old format ────────────────────────
        println!("╔═══ FAFB receptor_types test ═══╗");
        println!();
        let fafb_path = data("fafb.taas");
        println!("1. Loading {} (old format) ...", fafb_path.display());
        let mut f = File::open(&fafb_path).expect("fafb.taas not found");
        let mut conn = ConnectomeData::load_baked(&mut f).unwrap();
        drop(f);

        let n = conn.n_neurons;
        let n_syn = conn.n_synapses;
        let e_count = conn.nt_types.iter().filter(|&&t| t > 0.0).count();
        let i_count = conn.nt_types.iter().filter(|&&t| t < 0.0).count();
        println!("   ✓ Loaded: {n} neurons, {n_syn} synapses");
        println!("   ✓ NT types: E={e_count}, I={i_count}");
        println!("   ✓ receptor_types: {} (old format → empty)", conn.receptor_types.len());
        assert!(conn.receptor_types.is_empty(), "old format should have empty receptor_types");
        println!("   ✓ Backward compat: OK");

        // ─── 2. Build receptor_types ───────────────────
        println!();
        println!("2. Building receptor_types from nt_types ...");
        conn.build_receptor_types();
        let rt = &conn.receptor_types;
        assert_eq!(rt.len(), n_syn, "receptor_types len must match n_synapses");

        let ampa = rt.iter().filter(|&&r| r == 0).count();
        let gaba_a = rt.iter().filter(|&&r| r == 2).count();
        let nmda = rt.iter().filter(|&&r| r == 1).count();
        let gaba_b = rt.iter().filter(|&&r| r == 3).count();
        println!("   ✓ AMPA (0):   {ampa:>10} ({:.1}%)", ampa as f64 / n_syn as f64 * 100.0);
        println!("   ✓ GABA_A (2): {gaba_a:>10} ({:.1}%)", gaba_a as f64 / n_syn as f64 * 100.0);
        if nmda > 0 {
            println!("   ✓ NMDA (1):   {nmda:>10}");
        }
        if gaba_b > 0 {
            println!("   ✓ GABA_B (3): {gaba_b:>10}");
        }
        assert!(ampa > 0, "should have AMPA synapses");
        assert!(gaba_a > 0, "should have GABA_A synapses");
        assert_eq!(ampa + gaba_a + nmda + gaba_b, n_syn, "all synapses accounted for");
        println!();
        println!("   build_receptor_types(): OK");

        // ─── 3. Save to new format ─────────────────────
        let new_path = data("fafb_rt.taas");
        println!("3. Saving to new format ({}) ...", new_path.display());
        let mut out = File::create(&new_path).expect("create fafb_rt.taas");
        conn.save_baked(&mut out).unwrap();
        drop(out);

        let file_size = std::fs::metadata(&new_path).unwrap().len();
        println!("   ✓ Saved: {:.1}MB", file_size as f64 / 1_048_576.0);
        assert!(file_size > 0, "file should not be empty");

        // ─── 4. Load back and verify roundtrip ─────────
        println!("4. Loading back (roundtrip verification) ...");
        let mut f2 = File::open(&new_path).expect("open fafb_rt.taas");
        let loaded = ConnectomeData::load_baked(&mut f2).unwrap();
        drop(f2);

        assert_eq!(loaded.n_neurons, n, "n_neurons mismatch");
        assert_eq!(loaded.n_synapses, n_syn, "n_synapses mismatch");
        assert_eq!(loaded.values.len(), n_syn, "values length mismatch");
        assert_eq!(loaded.nt_types.len(), n, "nt_types length mismatch");
        assert_eq!(loaded.receptor_types.len(), n_syn, "receptor_types length after load");
        assert_eq!(loaded.receptor_types, conn.receptor_types, "receptor_types content mismatch");
        assert_eq!(loaded.values, conn.values, "values content mismatch");
        assert_eq!(loaded.nt_types, conn.nt_types, "nt_types content mismatch");
        assert_eq!(loaded.row_ptr, conn.row_ptr, "row_ptr mismatch");
        assert_eq!(loaded.col_indices, conn.col_indices, "col_indices mismatch");

        let amp = loaded.receptor_types.iter().filter(|&&r| r == 0).count();
        let gab = loaded.receptor_types.iter().filter(|&&r| r == 2).count();
        assert_eq!(amp, ampa, "AMPA count mismatch after roundtrip");
        assert_eq!(gab, gaba_a, "GABA_A count mismatch after roundtrip");
        println!("   ✓ Roundtrip: all arrays match");
        println!("   ✓ receptor_types preserved: AMPA={amp}, GABA_A={gab}");

        // ─── Summary ───────────────────────────────────
        println!();
        println!("╔═══ Summary ═══════════════════════════════════╗");
        println!("║  File:        {}", fafb_path.display());
        println!("║  New format:  {}", new_path.display());
        println!("║  Size:        {:.1}MB", file_size as f64 / 1_048_576.0);
        println!("║  Neurons:     {n}");
        println!("║  Synapses:    {n_syn}");
        println!("║  E/I:         {e_count}E / {i_count}I");
        println!("║  AMPA:        {ampa} ({:.1}%)", ampa as f64 / n_syn as f64 * 100.0);
        println!("║  GABA_A:      {gaba_a} ({:.1}%)", gaba_a as f64 / n_syn as f64 * 100.0);
        println!("║  Time:        {:.1}s", t0.elapsed().as_secs_f64());
        println!("╚═══════════════════════════════════════════════╝");
    }
}

#[cfg(not(feature = "connectome"))]
mod inner {
    pub fn run() {
        println!("This example requires the `connectome` feature.");
        println!("  cargo run --features connectome --example fafb_nt --release");
    }
}

fn main() {
    inner::run();
}
