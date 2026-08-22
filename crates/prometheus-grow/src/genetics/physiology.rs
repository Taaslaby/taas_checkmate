//! DNA → physiology: gene_physiology.py port.
//! NT-gene → sign, ion-channel level → HAAP (17 cols). Same constants as Python.

pub const N_HAAP: usize = 17;
pub const DT: f32 = 0.05;

/// NT → (gene, sign). Single source mirrors gene_physiology.py NT_GENE:
/// +1 excitatory, -1 inhibitory, 0 neuromodulator. Glutamate is INHIBITORY
/// in Drosophila (GluCl, Cl⁻) — unlike mammals.
pub fn nt_gene(nt: &str) -> (&'static str, f32) {
    match nt {
        "gaba" => ("Gad1", -1.0),
        "acetylcholine" => ("ChAT", 1.0),
        "glutamate" => ("VGlut", -1.0),
        "octopamine" => ("Tbh", 0.0),
        "dopamine" => ("DAT", 0.0),
        "serotonin" => ("Trh", 0.0),
        "histamine" => ("HDC", 0.0),
        _ => ("?", 0.0),
    }
}

/// channel → (col, lo, hi) specs, level∈[0,1] → lo+level*(hi-lo)
pub struct ChanSpec { pub col: usize, pub lo: f32, pub hi: f32 }
pub fn ion_channels(gene: &str) -> &'static [ChanSpec] {
    match gene {
        "HCN" => &[ChanSpec{col:6, lo:0.10, hi:0.60}, ChanSpec{col:9, lo:1.0/400.0, hi:1.0/150.0}, ChanSpec{col:12, lo:-80.0, hi:-65.0}],
        "KCNQ" => &[ChanSpec{col:4, lo:0.20, hi:0.60}, ChanSpec{col:8, lo:1.0/150.0, hi:1.0/60.0}, ChanSpec{col:16, lo:0.010, hi:0.050}],
        "para" => &[ChanSpec{col:14, lo:-45.0, hi:-35.0}],
        "leak" => &[ChanSpec{col:2, lo:0.08, hi:0.15}, ChanSpec{col:3, lo:-70.0, hi:-48.0}],
        _ => &[],
    }
}

pub fn fixed() -> [(usize,f32);9] {
    [(0, DT),(1,1.0),(5,-80.0),(7,-30.0),(10,-35.0),(11,5.0),(13,6.0),(15,-55.0),(0,DT)]
}

pub fn class_levels(cls: &str, nt: &str) -> [(&'static str,f32);4] {
    let mut lv = match cls {
        "motor neuron" => [("HCN",0.35),("KCNQ",0.90),("para",0.80),("leak",0.30)],
        "intrinsic neuron" => [("HCN",0.55),("KCNQ",0.60),("para",0.45),("leak",0.45)],
        "sensory neuron" => [("HCN",0.60),("KCNQ",0.40),("para",0.70),("leak",0.75)],
        "descending neuron" => [("HCN",0.35),("KCNQ",0.85),("para",0.70),("leak",0.25)],
        "ascending neuron" => [("HCN",0.50),("KCNQ",0.60),("para",0.45),("leak",0.45)],
        "sensory ascending" => [("HCN",0.60),("KCNQ",0.40),("para",0.70),("leak",0.75)],
        "efferent neuron" => [("HCN",0.35),("KCNQ",0.90),("para",0.80),("leak",0.30)],
        _ => [("HCN",0.45),("KCNQ",0.55),("para",0.50),("leak",0.40)],
    };
    if nt=="gaba" { for (g,v) in &mut lv { if *g=="HCN" { *v=(*v+0.15f32).min(1.0); } } }
    lv
}

pub fn gene_haap(cls: &str, nt: &str) -> [f32; N_HAAP] {
    let mut row=[0.0f32; N_HAAP];
    for (c,v) in [(0,DT),(1,1.0),(5,-80.0),(7,-30.0),(10,-35.0),(11,5.0),(13,6.0),(15,-55.0)] { row[c]=v; }
    let lv=class_levels(cls, nt);
    for (gene,lvl) in lv {
        for s in ion_channels(gene) { row[s.col]=s.lo+lvl*(s.hi-s.lo); }
    }
    row
}
