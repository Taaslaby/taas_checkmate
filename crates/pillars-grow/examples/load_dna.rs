//! Load the raw *Drosophila* genome (FASTA + GFF) and pull out real gene DNA.
//!
//! For each gene we want (NT identity, morphogens, axon guidance) this reads
//! its coordinates from the GFF and its A/C/G/T from the FASTA, then feeds the
//! sequences into a [`FlyGenome`] so the growth genome carries the real DNA,
//! not just gene names.
//!
//! ```bash
//! cargo run -p pillars-grow --example load_dna -- \
//!     C:/Users/anton/Desktop/Проекты/PROMETHEUS/drosophila_genome.fna.gz \
//!     C:/Users/anton/Desktop/Проекты/PROMETHEUS/drosophila_genes.gff.gz
//! ```

use std::collections::HashMap;
use std::path::Path;

use pillars_grow::{extract_gene, parse_fasta, parse_gene_coords, read_maybe_gzip, FlyGenome};

fn main() {
    let fa_path = std::env::args().nth(1).unwrap_or_else(|| {
        "C:/Users/anton/Desktop/Проекты/PROMETHEUS/drosophila_genome.fna.gz".into()
    });
    let gff_path = std::env::args().nth(2).unwrap_or_else(|| {
        "C:/Users/anton/Desktop/Проекты/PROMETHEUS/drosophila_genes.gff.gz".into()
    });
    assert!(Path::new(&fa_path).exists(), "нет FASTA: {fa_path}");
    assert!(Path::new(&gff_path).exists(), "нет GFF: {gff_path}");

    println!("=== загрузка генома мухи ===");
    let fa = read_maybe_gzip(&fa_path).expect("не читается FASTA");
    let seqs = parse_fasta(&fa);
    let total_bp: usize = seqs.values().map(|s| s.len()).sum();
    println!(
        "FASTA: {} записей, {:.0} Mb A/C/G/T",
        seqs.len(),
        total_bp as f64 / 1e6
    );

    let gff = read_maybe_gzip(&gff_path).expect("не читается GFF");
    // NT identity + morphogens + axon guidance — the "genome" of a neuron.
    let wanted = [
        "Gad1", "ChAT", "VGlut", "VGAT", "ple", // neurotransmitters
        "dpp", "wg", "hh", // morphogens
        "fra", "comm", "robo", // axon guidance
    ];
    let coords = parse_gene_coords(&gff, &wanted);
    println!(
        "GFF: найдено {}/{} целевых генов\n",
        coords.len(),
        wanted.len()
    );

    let mut extracted: HashMap<String, String> = HashMap::new();
    for sym in &wanted {
        match coords.get(*sym) {
            Some(c) => {
                let seq = extract_gene(&seqs, c);
                let head: String = seq.chars().take(60).collect();
                println!(
                    "{:<6} {}:{}..{} ({})  len={}  {}...",
                    sym,
                    c.0,
                    c.1,
                    c.2,
                    c.3,
                    seq.len(),
                    head
                );
                extracted.insert(sym.to_string(), seq);
            }
            None => println!("{sym:<6} — не найдено в GFF"),
        }
    }

    // Build the half-center genome carrying real DNA sequences.
    let genome = FlyGenome::half_center().with_sequences(&extracted);
    println!("\n=== FlyGenome (с реальной ДНК) ===");
    println!(
        "  nt_gene      = {} ({} bp)",
        genome.nt_gene,
        seq_len(&genome.nt_gene_seq)
    );
    println!(
        "  morphogen    = {} ({} bp)",
        genome.morphogen_gene,
        seq_len(&genome.morphogen_gene_seq)
    );
    println!(
        "  guidance     = {} ({} bp)",
        genome.guidance_gene,
        seq_len(&genome.guidance_gene_seq)
    );

    if let Some(s) = &genome.nt_gene_seq {
        println!("\n  Gad1 (GABA-синтез) первые 360 букв реальной ДНК:");
        for i in (0..s.len().min(360)).step_by(60) {
            println!("    {:>6}  {}", i + 1, &s[i..(i + 60).min(s.len())]);
        }
    }
}

fn seq_len(s: &Option<String>) -> usize {
    s.as_ref().map(|s| s.len()).unwrap_or(0)
}
