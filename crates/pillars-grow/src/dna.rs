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
            buf.push_str(&line.trim().to_uppercase());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fasta_handles_multi_record_and_whitespace() {
        let text = ">chr2L extra description\nacgtACGT\nNNnn\n>chr3\nTTTT\n>chr4\n";
        let seqs = parse_fasta(text);
        assert_eq!(seqs.len(), 3);
        // Accession is the first token; bases are uppercased, whitespace stripped.
        assert_eq!(seqs["chr2L"], "ACGTACGTNNNN");
        assert_eq!(seqs["chr3"], "TTTT");
        assert!(seqs.contains_key("chr4"));
        assert_eq!(seqs["chr4"], "");
    }

    #[test]
    fn parse_fasta_empty_input() {
        assert!(parse_fasta("").is_empty());
        assert!(parse_fasta(">only_header\n").len() == 1);
    }

    #[test]
    fn revcomp_complements_and_reverses() {
        assert_eq!(revcomp("ATGC"), "GCAT");
        assert_eq!(revcomp("ATGC"), revcomp("ATGC")); // palindromic-ish check
        assert_eq!(revcomp("AATT"), "AATT");
        assert_eq!(revcomp("acgt"), "ACGT");
        // Unknown bases map to N.
        assert_eq!(revcomp("AXG"), "CNT");
    }

    #[test]
    fn parse_gene_coords_prefers_gene_attr_and_dedups() {
        let gff = [
            "##gff-version 3",
            "2R\tFlyBase\tgene\t1000\t2000\t.\t+\t.\tID=FBgn1;gene=eve;Name=even-skipped",
            // Second eve line — must be ignored (first wins).
            "2R\tFlyBase\tgene\t9000\t9500\t.\t-\t.\tID=FBgn9;gene=eve;Name=other",
            "2R\tFlyBase\tgene\t3000\t4000\t.\t-\t.\tID=FBgn2;Name=ftz",
            "2R\tFlyBase\texon\t500\t600\t.\t+\t.\tgene=not-a-gene",
            "2R\tFlyBase\tgene\t5000\t6000\t.\t-\t.\tID=FBgn3;gene=hb",
        ]
        .join("\n");
        let coords = parse_gene_coords(&gff, &["eve", "ftz", "hb"]);
        assert_eq!(coords.len(), 3);
        // gene= beats Name=; first occurrence of `eve` wins (the + strand one).
        assert_eq!(coords["eve"], ("2R".to_string(), 1000, 2000, '+'));
        // Name= fallback for `ftz`.
        assert_eq!(coords["ftz"], ("2R".to_string(), 3000, 4000, '-'));
        assert_eq!(coords["hb"], ("2R".to_string(), 5000, 6000, '-'));
        // Unwanted symbol never appears.
        assert!(parse_gene_coords(&gff, &["none"]).is_empty());
    }

    #[test]
    fn extract_gene_slices_and_revcomps() {
        let mut seqs = HashMap::new();
        seqs.insert("2R".to_string(), "AACCGGTTAACCGGTT".to_string());

        // 1-based inclusive coordinates: positions 3..=6 → "CCGG".
        let fwd = ("2R".to_string(), 3usize, 6usize, '+');
        assert_eq!(extract_gene(&seqs, &fwd), "CCGG");

        // Minus strand: revcomp of "CCGG" is "CCGG"; use an asymmetric window.
        let minus = ("2R".to_string(), 5usize, 8usize, '-');
        // Window 5..=8 of "AACCGGTTAACCGGTT": chars 5,6,7,8 = "GGTT".
        assert_eq!(revcomp("GGTT"), "AACC");
        assert_eq!(extract_gene(&seqs, &minus), "AACC");

        // Unknown chromosome → empty.
        let missing = ("3R".to_string(), 1usize, 4usize, '+');
        assert_eq!(extract_gene(&seqs, &missing), "");
    }

    #[test]
    fn read_maybe_gzip_transparently_decompresses() {
        let dir = std::env::temp_dir();
        let plain = dir.join(format!("pillars_grow_plain_{}.txt", std::process::id()));
        let gz = dir.join(format!("pillars_grow_gz_{}.gz", std::process::id()));
        std::fs::write(&plain, "hello\n").unwrap();

        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, b"hello gz").unwrap();
        let bytes = enc.finish().unwrap();
        std::fs::write(&gz, bytes).unwrap();

        assert_eq!(read_maybe_gzip(plain.to_str().unwrap()).unwrap(), "hello\n");
        assert_eq!(read_maybe_gzip(gz.to_str().unwrap()).unwrap(), "hello gz");

        let _ = std::fs::remove_file(&plain);
        let _ = std::fs::remove_file(&gz);
    }
}
