//! Load the raw *Drosophila* genome: FASTA sequences + GFF gene coordinates,
//! and extract a gene's DNA (reverse-complemented on the minus strand).
//!
//! This is the Rust counterpart of PROMETHEUS `dna_nt_map.py`'s
//! `load_genome` / `load_gene_coords` / `revcomp`, so the growth engine can
//! hold the real A/C/G/T of each gene, not just its name.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use flate2::read::GzDecoder;

/// Read a text file, transparently decompressing `.gz`.
pub fn read_maybe_gzip(path: &str) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut s = String::new();
    if path.ends_with(".gz") {
        GzDecoder::new(file).read_to_string(&mut s)?;
    } else {
        let mut f = file;
        f.read_to_string(&mut s)?;
    }
    Ok(s)
}

/// Parse a FASTA file: accession → sequence (uppercase, whitespace stripped).
pub fn parse_fasta(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut cur: Option<String> = None;
    let mut buf = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(acc) = cur.take() {
                out.insert(acc, std::mem::take(&mut buf));
            }
            let acc = rest.split_whitespace().next().unwrap_or("").to_string();
            cur = Some(acc);
        } else if cur.is_some() {
            buf.push_str(line.trim());
        }
    }
    if let Some(acc) = cur {
        out.insert(acc, buf);
    }
    out
}

/// Gene coordinates for the wanted symbols: `symbol → (chrom, start, end, strand)`.
///
/// Reads GFF3 `gene` lines and resolves the symbol from the attributes
/// (`gene=` with `Name=` fallback, matching `dna_nt_map.py`). Coordinates are
/// 1-based (as in GFF); `start..end` maps to Python's `seq[start-1..end]`.
pub fn parse_gene_coords(
    gff: &str,
    wanted: &[&str],
) -> HashMap<String, (String, usize, usize, char)> {
    let mut out = HashMap::new();
    for line in gff.lines() {
        if line.starts_with('#') {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 9 || c[2] != "gene" {
            continue;
        }
        let attr = c[8];
        let mut name: Option<&str> = None;
        let mut gene: Option<&str> = None;
        for kv in attr.split(';') {
            if let Some(v) = kv.strip_prefix("Name=") {
                name = Some(v);
            } else if let Some(v) = kv.strip_prefix("gene=") {
                gene = Some(v);
            }
        }
        let sym = gene.or(name).unwrap_or("");
        if wanted.contains(&sym) && !out.contains_key(sym) {
            let start: usize = c[3].parse().unwrap_or(0);
            let end: usize = c[4].parse().unwrap_or(0);
            let strand = c[6].chars().next().unwrap_or('+');
            out.insert(sym.to_string(), (c[0].to_string(), start, end, strand));
        }
    }
    out
}

/// Reverse-complement a DNA string (N maps to N).
pub fn revcomp(s: &str) -> String {
    s.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'C' | 'c' => 'G',
            'G' | 'g' => 'C',
            _ => 'N',
        })
        .collect()
}

/// Extract a gene's DNA sequence from the chromosome sequence, applying the
/// reverse complement on the minus strand.
pub fn extract_gene(
    seqs: &HashMap<String, String>,
    coords: &(String, usize, usize, char),
) -> String {
    let (chrom, start, end, strand) = coords;
    let Some(full) = seqs.get(chrom) else {
        return String::new();
    };
    let s = full
        .get(start.saturating_sub(1)..*end)
        .unwrap_or("")
        .to_string();
    if *strand == '-' {
        revcomp(&s)
    } else {
        s
    }
}
