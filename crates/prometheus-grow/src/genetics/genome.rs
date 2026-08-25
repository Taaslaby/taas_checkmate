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
/// Columns: `ген,нейротрансмиттер,connkit,знак,роль,хромосома,start,end,strand,длина(ДНК),biotype`
/// (column names are Russian in the source CSV and must match it verbatim).
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

/// One row of the DNA → channel map: a gene, its family, and the decoder
/// parameter it drives (`"HCN"`, `"KCNQ"`, `"para"`, `"leak"`, … or
/// `"cell-recognition"` for the adhesion genes that set wiring).
#[derive(Clone, Debug)]
pub struct ChannelGene {
    pub gene: String,
    pub family: String,
    /// Decoder parameter: `"HCN"`, `"KCNQ"`, `"Shaker"`, `"para"`, `"SK"`,
    /// `"Ca"`, `"leak"`, or `"cell-recognition"`.
    pub param: String,
}

/// Parse the `dna_channel_map.csv` produced by PROMETHEUS.
///
/// Columns: `ген,семейство,параметр декодера,HAAP/что задаёт,роль,…`
/// (column names are Russian in the source CSV and must match it verbatim).
/// We only need `ген` (0), `семейство` (1) and `параметр декодера` (2);
/// the later columns may contain quoted commas and are not parsed.
pub fn parse_channel_map(csv: &str) -> Vec<ChannelGene> {
    let mut out = Vec::new();
    for line in csv.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f = split_csv(line);
        if f.len() < 3 {
            continue;
        }
        out.push(ChannelGene {
            gene: f[0].trim().to_string(),
            family: f[1].trim().to_string(),
            param: f[2].trim().to_string(),
        });
    }
    out
}

/// Split a CSV line into fields, honouring double-quoted fields so quoted
/// commas (e.g. `"SK (KCa, small)"`) do not split the field.
fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    fields.push(cur);
    fields
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
    /// Cell-recognition (adhesion) gene driving the wiring rule (e.g. `"Dscam1"`).
    pub adhesion_gene: Option<String>,
    /// Ion-channel genes this cell expresses (set dynamics, not topology).
    pub channel_genes: Vec<String>,
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
            adhesion_gene: Some("Dscam1".into()),
            channel_genes: vec!["Ih".into(), "KCNQ".into()],
        }
    }

    /// Attach the raw DNA sequences (gene → A/C/G/T) loaded from FASTA/GFF.
    pub fn with_sequences(mut self, seqs: &std::collections::HashMap<String, String>) -> Self {
        self.nt_gene_seq = seqs.get(&self.nt_gene).cloned();
        self.morphogen_gene_seq = seqs.get(&self.morphogen_gene).cloned();
        self.guidance_gene_seq = seqs.get(&self.guidance_gene).cloned();
        self
    }

    /// Resolve the cell type from the DNA map (gene → sign).
    pub fn cell_type(&self, map: &[DnaGene]) -> Option<CellType> {
        map.iter().find(|g| g.gene == self.nt_gene).map(|g| {
            if g.sign < 0 {
                CellType::Inhibitory
            } else {
                CellType::Excitatory
            }
        })
    }

    /// Human-readable name of the neurotransmitter this genome's cells make.
    pub fn neurotransmitter(&self, map: &[DnaGene]) -> Option<String> {
        map.iter()
            .find(|g| g.gene == self.nt_gene)
            .map(|g| g.neurotransmitter.clone())
    }

    /// Resolve the wiring rule from the adhesion gene: a cell-recognition
    /// gene (Dscam1/CadN/Tl) implies isoform-specific same-type recognition
    /// (the DNA → topology link); no adhesion gene → unrestricted wiring.
    pub fn recognition(&self, channel_map: &[ChannelGene]) -> crate::Recognition {
        let has = self.adhesion_gene.as_deref().is_some_and(|g| {
            channel_map
                .iter()
                .any(|c| c.gene == g && c.param == "cell-recognition")
        });
        if has {
            crate::Recognition::same_type()
        } else {
            crate::Recognition::all()
        }
    }

    /// The channel genes this genome expresses, resolved from the channel
    /// map. Channels set dynamics (rebound, adaptation, excitability), not
    /// wiring — see the R²≈0.003 result (DNA does not encode synaptic strength).
    pub fn channels(&self, channel_map: &[ChannelGene]) -> Vec<ChannelGene> {
        channel_map
            .iter()
            .filter(|c| self.channel_genes.iter().any(|g| g == &c.gene))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL_CSV: &str = "\
ген,семейство,параметр декодера,HAAP/что задаёт,роль,хромосома,start,end,strand,длина(ДНК),biotype\n\
Ih,HCN-канал,HCN,\"g_h, tau_h\",post-inhibitory rebound (half-center),NT_033778.4,14276130,14300095,+,23966,protein_coding\n\
Dscam1,Dscam1 (адгезия),cell-recognition,conn_prob,клеточное распознавание → проводка,NC_004354.4,1,100,+,99,protein_coding\n";

    #[test]
    fn parses_channel_map() {
        let map = parse_channel_map(CHANNEL_CSV);
        assert_eq!(map.len(), 2);
        assert_eq!(map[0].gene, "Ih");
        assert_eq!(map[0].param, "HCN");
        assert_eq!(map[1].param, "cell-recognition");
    }

    #[test]
    fn adhesion_gene_enables_same_type_recognition() {
        let map = parse_channel_map(CHANNEL_CSV);
        let mut g = FlyGenome::half_center();
        g.adhesion_gene = Some("Dscam1".into());
        let rec = g.recognition(&map);
        assert!(rec.allowed[1][1], "inh→inh разрешён при Dscam1");
        assert!(!rec.allowed[0][1], "cross-type запрещён при Dscam1");
        // no adhesion — unrestricted
        g.adhesion_gene = None;
        let rec2 = g.recognition(&map);
        assert!(rec2.allowed[0][1] && rec2.allowed[1][0]);
    }

    #[test]
    fn parses_quoted_family_field() {
        let csv = "gene,family,param\nSK,\"SK (KCa, small)\",SK\n";
        let map = parse_channel_map(csv);
        assert_eq!(map.len(), 1);
        assert_eq!(map[0].family, "SK (KCa, small)");
        assert_eq!(map[0].param, "SK");
    }
}
