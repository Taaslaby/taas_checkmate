/// Конвертация старого .taas (SAAT1, `ConnectomeData::save_baked`) → новый TAAS1.
///
/// Запуск:
///   cargo run --features connectome --release --example convert_taas_format -- \
///       fafb_15m.taas fafb_15m_new.taas
use std::fs::File;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: convert_taas_format <input.taas> <output.taas>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    println!("📖 Loading old-format .taas: {input} ...");
    let t0 = Instant::now();

    let mut f = File::open(input).expect("Cannot open input file");
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f)
        .expect("Failed to load old-format connectome");
    drop(f);

    println!(
        "   ✓ {n} neurons, {s} synapses in {t:.2}s",
        n = conn.n_neurons,
        s = conn.n_synapses,
        t = t0.elapsed().as_secs_f64()
    );
    println!(
        "   Groups: {:?}",
        conn.groups.iter().map(|g| &g.name).collect::<Vec<_>>()
    );

    let t1 = Instant::now();
    let mut writer = taas_format::TaasWriter::new();
    writer.add_metadata("converted_from", "old_baked_format");
    writer.add_metadata("original_neurons", &conn.n_neurons.to_string());
    writer.add_metadata("original_synapses", &conn.n_synapses.to_string());
    taas_format::save_connectome(&conn, &mut writer);

    writer
        .save_to_file(output)
        .expect("Failed to save new-format connectome");
    println!("   ✓ Saved new TAAS1 format: {output}");
    println!(
        "   📦 New format: {}MB in {:.1}s",
        std::fs::metadata(output)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0),
        t1.elapsed().as_secs_f64()
    );
    println!("   🧬 Ready for taas_format::load_connectome() / Python bindings");
}
