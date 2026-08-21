//! The fly-genome bridge: load real *Drosophila* gene data (the DNA → NT map
//! produced by PROMETHEUS) and use it to instantiate a growth genome.
//!
//! The growth engine's genome is normally abstract (`CellType::Inhibitory`,
//! morphogen `"A"`). Here those fields are resolved from real genes:
//!
//! ```text
//! dna_nt_map.csv:  gene → neurotransmitter → sign
//!                    │
//!                    ▼
//! FlyGenome { nt_gene: "Gad1", morphogen_gene: "dpp", guidance_gene: "fra" }
//!                    │
//!                    ▼
//! Gad1 → sign -1 → CellType::Inhibitory   (the half-center cell identity)
//! ```

use crate::cell::CellType;

/// One row of the DNA → NT map: a gene, its neurotransmitter, and the synapse
/// sign it produces (`+1` excitatory, `-1` inhibitory).
#[derive(Clone, Debug)]
pub struct DnaGene {
    pub gene: String,
    pub neurotransmitter: String,
    pub sign: i8,
}

/// Parse the `dna_nt_map.csv` produced by PROMETHEUS.
///
/// Columns: `ген,нейротрансмиттер,connkit,знак,роль,хромосома,start,end,strand,длина(ДНК),biotype`.
/// We only need `ген` (0), `нейротрансмиттер` (1) and `знак` (3).
pub fn parse_dna_map(csv: &str) -> Vec<DnaGene> {
    let mut out = Vec::new();
    for line in csv.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let sign: i8 = f[3].trim().parse().unwrap_or(0);
        out.push(DnaGene {
            gene: f[0].trim().to_string(),
            neurotransmitter: f[1].trim().to_string(),
            sign,
        });
    }
    out
}

/// A growth genome instantiated from real gene names.
#[derive(Clone, Debug)]
pub struct FlyGenome {
    /// Neurotransmitter-synthesis gene that sets the cell's sign (e.g. `"Gad1"`).
    pub nt_gene: String,
    /// Morphogen gene the cells secrete (e.g. `"dpp"`).
    pub morphogen_gene: String,
    /// Axon-guidance receptor gene the growth cone expresses (e.g. `"fra"`).
    pub guidance_gene: String,
    /// Raw DNA sequence of the NT gene (loaded from FASTA/GFF, if any).
    pub nt_gene_seq: Option<String>,
    /// Raw DNA sequence of the morphogen gene (loaded from FASTA/GFF, if any).
    pub morphogen_gene_seq: Option<String>,
    /// Raw DNA sequence of the guidance gene (loaded from FASTA/GFF, if any).
    pub guidance_gene_seq: Option<String>,
}

impl FlyGenome {
    /// The half-center genome: GABA identity + dpp morphogen + fra guidance.
    pub fn half_center() -> Self {
        FlyGenome {
            nt_gene: "Gad1".into(),
            morphogen_gene: "dpp".into(),
            guidance_gene: "fra".into(),
            nt_gene_seq: None,
            morphogen_gene_seq: None,
            guidance_gene_seq: None,
        }
    }

    /// Attach the raw DNA sequences (gene → A/C/G/T) loaded from FASTA/GFF.
    pub fn with_sequences(
        mut self,
        seqs: &std::collections::HashMap<String, String>,
    ) -> Self {
        self.nt_gene_seq = seqs.get(&self.nt_gene).cloned();
        self.morphogen_gene_seq = seqs.get(&self.morphogen_gene).cloned();
        self.guidance_gene_seq = seqs.get(&self.guidance_gene).cloned();
        self
    }

    /// Resolve the cell type from the DNA map (gene → sign).
    pub fn cell_type(&self, map: &[DnaGene]) -> Option<CellType> {
        map.iter()
            .find(|g| g.gene == self.nt_gene)
            .map(|g| if g.sign < 0 {
                CellType::Inhibitory
            } else {
                CellType::Excitatory
            })
    }

    /// Human-readable name of the neurotransmitter this genome's cells make.
    pub fn neurotransmitter(&self, map: &[DnaGene]) -> Option<String> {
        map.iter()
            .find(|g| g.gene == self.nt_gene)
            .map(|g| g.neurotransmitter.clone())
    }
}
