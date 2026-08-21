//! Universal connectome data container.
//!
//! [`ConnectomeData`] holds the CSR (Compressed Sparse Row) representation
//! of any directed graph with signed weights (excitatory/inhibitory).
//!
//! Use it to build a connectome from any source (FlyWire, MANC, synthetic, etc.)
//! and then pass it to [`BrainRunner`](crate::runner::BrainRunner) for simulation.

use std::io::{self, Read, Write};

/// .taas baked connectome magic bytes ("TAAS").
const TAAS_MAGIC: u32 = 0x54414153;
/// Current .taas format version.
const TAAS_VERSION: u32 = 1;

/// Cast any POD slice to &[u8] for raw I/O.
unsafe fn pod_as_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
}

/// A named group of neuron indices (e.g. "dn", "mn", "visual_afferents").
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuronGroup {
    pub name: String,
    pub indices: Vec<usize>,
}

/// Universal connectome: CSR matrix + NT types + typed neuron groups.
///
/// # Example
///
/// ```ignore
/// let conn = ConnectomeData {
///     n_neurons: 108247,
///     n_synapses: 3730122,
///     row_ptr: vec![...],
///     col_indices: vec![...],
///     values: vec![...],
///     nt_types: vec![...],   // +1 exc, -1 inh
///     groups: vec![
///         NeuronGroup { name: "dn".into(), indices: dn_list.clone() },
///         NeuronGroup { name: "mn".into(), indices: mn_list.clone() },
///     ],
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConnectomeData {
    /// Number of neurons in the graph.
    pub n_neurons: usize,
    /// Number of synapses (non-zero entries in CSR).
    pub n_synapses: usize,
    /// CSR row pointer [n_neurons + 1].
    pub row_ptr: Vec<i32>,
    /// CSR column indices [n_synapses].
    pub col_indices: Vec<i32>,
    /// Synaptic weights [n_synapses].
    pub values: Vec<f32>,
    /// Neurotransmitter type per pre-neuron: +1 excitatory, -1 inhibitory [n_neurons].
    pub nt_types: Vec<f32>,
    /// Receptor type per synapse (optional): 0=AMPA, 1=NMDA, 2=GABA_A, 3=GABA_B,
    /// 4=nAChR, 5=GluCl, 6=mAChR.
    /// If empty, synaptic kinetics uses simple instant-current (nt_types only).
    pub receptor_types: Vec<u8>,
    /// Named neuron groups (e.g. "dn", "mn", "visual").
    pub groups: Vec<NeuronGroup>,
}

impl ConnectomeData {
    /// Create a new connectome from raw CSR arrays.
    ///
    /// `groups` should be a list of named groups: `vec![("dn", dn_indices), ("mn", mn_indices)]`
    pub fn new(
        row_ptr: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<f32>,
        nt_types: Vec<f32>,
        groups: Vec<(&str, Vec<usize>)>,
    ) -> Self {
        let n_neurons = if row_ptr.len() > 1 {
            row_ptr.len() - 1
        } else {
            0
        };
        let n_synapses = values.len();

        assert_eq!(
            row_ptr.len(),
            n_neurons + 1,
            "row_ptr must have n_neurons + 1 elements"
        );
        assert_eq!(
            col_indices.len(),
            n_synapses,
            "col_indices length must match values length"
        );
        assert!(
            nt_types.is_empty() || nt_types.len() >= n_neurons,
            "nt_types must have at least n_neurons elements"
        );

        Self {
            n_neurons,
            n_synapses,
            row_ptr,
            col_indices,
            values,
            nt_types: if nt_types.is_empty() {
                vec![1.0; n_neurons]
            } else {
                nt_types
            },
            receptor_types: Vec::new(),
            groups: groups
                .into_iter()
                .map(|(name, indices)| NeuronGroup {
                    name: name.to_string(),
                    indices,
                })
                .collect(),
        }
    }

    /// Get a named group by name. Returns None if not found.
    pub fn group(&self, name: &str) -> Option<&[usize]> {
        self.groups
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.indices.as_slice())
    }

    /// Build per-synapse receptor_types from nt_types (simple sign-based mapping).
    ///
    /// Uses `build_receptor_types_from_csr()` which maps NT sign to receptor:
    /// - `nt > 0` (excitatory) → AMPA (0)
    /// - `nt <= 0` (inhibitory) → GABA_A (2)
    ///
    /// If receptor_types is already non-empty, does nothing (preserves existing mapping).
    pub fn build_receptor_types(&mut self) {
        if !self.receptor_types.is_empty() {
            return; // already populated
        }
        self.receptor_types = crate::data::build_receptor_types_from_csr(
            &self.row_ptr,
            self.n_synapses,
            &self.nt_types,
        );
    }

    /// Build per-synapse receptor_types from textual NT labels (from neurons.csv.gz).
    ///
    /// Maps textual labels to Drosophila-appropriate receptors:
    /// - "ACh" → nAChR (4, fast exc)
    /// - "GABA" → GABA_A (2, fast inh)
    /// - "GLUT" → GluCl (5, fast inh in insects!)
    /// - "DA" / "SER" / "OCT" → mAChR (6, slow mod)
    ///
    /// If receptor_types is already non-empty, does nothing.
    pub fn build_receptor_types_from_csv(&mut self, nt_labels: &[String]) {
        if !self.receptor_types.is_empty() {
            return;
        }
        self.receptor_types =
            crate::data::build_receptor_types_from_csv(&self.row_ptr, self.n_synapses, nt_labels);
    }

    /// Build per-synapse receptor_types with FlyWire neurotransmitter mapping.
    ///
    /// Maps nt values to Drosophila-appropriate receptor types:
    /// | NT value | Interpretation | Receptor | Effect |
    /// |----------|----------------|----------|--------|
    /// | +1.0     | ACh            | nAChR (4) | Fast exc |
    /// | -1.0     | GABA           | GABA_A (2) | Fast inh |
    /// | +2.0     | GLUT           | GluCl (5) | **Inh** (Cl⁻) |
    /// | -2.0     | DA / SER / OCT | mAChR (6) | Slow mod |
    /// | 0.0      | None / gap     | AMPA (0) | Exc fallback |
    ///
    /// **Critical**: In insects, glutamate (GLUT) gates Cl⁻ channels (GluCl),
    /// making it **inhibitory**, not excitatory as in mammals.
    /// ACh (acetylcholine) is the main excitatory NT in Drosophila CNS.
    pub fn build_receptor_types_flywire(&mut self, nt_map: &[f32]) {
        if !self.receptor_types.is_empty() {
            return;
        }
        self.receptor_types = crate::data::build_receptor_types_from_csr_flywire(
            &self.row_ptr,
            self.n_synapses,
            nt_map,
        );
    }

    /// Переупорядочить коннектом через Reverse Cuthill-McKee (in-place).
    ///
    /// Модифицирует row_ptr, col_indices, values, nt_types и groups так,
    /// чтобы связанные нейроны были рядом в памяти. Это уменьшает cache misses
    /// при CSR propagate в 10-50×.
    ///
    /// Возвращает permutation: `perm[i] = new_index` для нейрона i.
    /// Используйте его чтобы переупорядочить state (v, u, I_buf) через
    /// `crate::rcm::apply_permutation_to_state()`.
    pub fn reorder_rcm(&mut self) -> Vec<usize> {
        let adj = crate::rcm::build_adjacency(self);
        let perm = crate::rcm::reverse_cuthill_mckee(&adj);

        let (new_rp, new_ci, new_vals, new_nt) = crate::rcm::apply_permutation_to_csr(
            &self.row_ptr,
            &self.col_indices,
            &self.values,
            &self.nt_types,
            &perm,
        );

        self.row_ptr = new_rp;
        self.col_indices = new_ci;
        self.values = new_vals;
        self.nt_types = new_nt;
        self.groups = crate::rcm::apply_permutation_to_groups(&self.groups, &perm);

        perm
    }

    /// Save to binary .taas format.
    ///
    /// Writes all arrays including receptor_types. Format:
    /// [magic(4) | version(4) | n_neurons(8) | n_synapses(8) |
    ///  row_ptr(4·(n+1)) | col_indices(4·s) | values(4·s) |
    ///  nt_types(4·n) | receptor_types(1·s) | has_receptors(1)]
    ///
    /// where `has_receptors` is a single byte: 1 if receptor_types is non-empty, 0 otherwise.
    /// If 1, the receptor_types array (n_synapses bytes) follows.
    ///
    /// Load via [`ConnectomeData::load_baked`].
    pub fn save_baked<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&TAAS_MAGIC.to_ne_bytes())?;
        w.write_all(&TAAS_VERSION.to_ne_bytes())?;
        w.write_all(&(self.n_neurons as u64).to_ne_bytes())?;
        w.write_all(&(self.n_synapses as u64).to_ne_bytes())?;

        unsafe {
            w.write_all(pod_as_bytes(&self.row_ptr))?;
            w.write_all(pod_as_bytes(&self.col_indices))?;
            w.write_all(pod_as_bytes(&self.values))?;
            w.write_all(pod_as_bytes(&self.nt_types))?;
        }

        // Write receptor_types
        let has_receptors = (!self.receptor_types.is_empty()) as u8;
        w.write_all(&[has_receptors])?;
        if has_receptors == 1 {
            unsafe {
                w.write_all(pod_as_bytes(&self.receptor_types))?;
            }
        }

        Ok(())
    }

    /// Load from binary .taas format.
    ///
    /// Validates magic + version, then reads arrays directly into
    /// properly aligned Vec allocations.
    pub fn load_baked<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf32 = [0u8; 4];
        let mut buf64 = [0u8; 8];

        r.read_exact(&mut buf32)?;
        let magic = u32::from_ne_bytes(buf32);
        if magic != TAAS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(".taas: bad magic {magic:#x}, expected {TAAS_MAGIC:#x}"),
            ));
        }

        r.read_exact(&mut buf32)?;
        let _version = u32::from_ne_bytes(buf32);

        r.read_exact(&mut buf64)?;
        let n_neurons = u64::from_ne_bytes(buf64) as usize;

        r.read_exact(&mut buf64)?;
        let n_synapses = u64::from_ne_bytes(buf64) as usize;

        // Read each array by writing directly into a properly aligned Vec.
        macro_rules! read_slice {
            ($n:expr, $ty:ty) => {{
                let mut v: Vec<$ty> = vec![0 as $ty; $n];
                let buf = unsafe {
                    std::slice::from_raw_parts_mut(
                        v.as_mut_ptr() as *mut u8,
                        $n * std::mem::size_of::<$ty>(),
                    )
                };
                r.read_exact(buf)?;
                v
            }};
        }

        Ok(Self {
            n_neurons,
            n_synapses,
            row_ptr: read_slice!(n_neurons + 1, i32),
            col_indices: read_slice!(n_synapses, i32),
            values: read_slice!(n_synapses, f32),
            nt_types: read_slice!(n_neurons, f32),
            // receptor_types is optional — written AFTER main arrays in save_baked
            receptor_types: {
                let mut rt = Vec::new();
                let mut has_receptors_buf = [0u8; 1];
                match r.read(&mut has_receptors_buf) {
                    Ok(1) if has_receptors_buf[0] == 1 => {
                        rt = read_slice!(n_synapses, u8);
                    }
                    _ => {} // EOF or has_receptors=0 → empty
                }
                rt
            },
            groups: Vec::new(),
        })
    }
}

/// Input channel — как внешний сигнал подаётся в мозг.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InputChannel {
    pub name: String,
    pub neuron_indices: Vec<usize>,
    pub scale: f32,
    pub encoding: InputEncoding,
}

/// Способ кодирования входного сигнала.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InputEncoding {
    /// Прямая инъекция тока: `I[neuron] += value * scale`
    Direct,
    /// Poisson-спайки с частотой `rate = value * scale`
    Poisson,
}

/// Output channel — как активность нейронов преобразуется в действие.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputChannel {
    pub name: String,
    pub neuron_indices: Vec<usize>,
    /// Число выходов (для linear readout). 0 = auto = len(neuron_indices)
    pub n_outputs: usize,
    pub readout: OutputReadout,
    /// Предобученные веса readout (optional, f32[n_outputs * len(neuron_indices)])
    pub weights: Option<Vec<f32>>,
}

/// Способ чтения выхода из нейронной активности.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OutputReadout {
    /// Сумма спайков за окно → linear layer
    Linear,
}

/// Полное описание I/O интерфейса мозга.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainIO {
    pub input_channels: Vec<InputChannel>,
    pub output_channels: Vec<OutputChannel>,
    /// Постоянный ток смещения на все нейроны
    pub bias: f32,
}

// ═══════════════════════════════════════════════════════════
//  Build receptor_types from CSR + nt_types
// ═══════════════════════════════════════════════════════════

/// Build per-synapse receptor_types from nt_types using CSR structure.
///
/// Maps: nt > 0 → AMPA (0), nt <= 0 → GABA_A (2).
/// Each synapse gets the receptor type of its pre-neuron's NT.
pub fn build_receptor_types_from_csr(
    row_ptr: &[i32],
    n_synapses: usize,
    nt_types: &[f32],
) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = if row_ptr.len() > 1 {
        row_ptr.len() - 1
    } else {
        0
    };
    let mut syn_idx = 0usize;
    for pre in 0..n {
        let nt = nt_types.get(pre).copied().unwrap_or(1.0);
        let rec = if nt > 0.0 { 0u8 } else { 2u8 }; // AMPA or GABA_A
        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(pre + 1).copied().unwrap_or(0) as usize).min(n_synapses);
        for _ in start..end {
            if syn_idx < receptors.len() {
                receptors[syn_idx] = rec;
            }
            syn_idx += 1;
        }
    }
    receptors
}

/// Build per-synapse receptor_types with Drosophila-appropriate FlyWire NT mapping.
///
/// | nt value | NT          | Receptor | Code | Effect |
/// |----------|-------------|----------|------|--------|
/// | +1.0     | ACh         | nAChR    | 4    | Fast exc |
/// | -1.0     | GABA        | GABA_A   | 2    | Fast inh |
/// | +2.0     | GLUT (Glu)  | GluCl    | 5    | **Inh** (Cl⁻) |
/// | -2.0     | DA/SER/OCT  | mAChR    | 6    | Slow mod |
/// | 0.0      | None/gap    | AMPA     | 0    | Exc fallback |
///
/// **In insects, glutamate is INHIBITORY** (GluCl, Cl⁻ channel),
/// unlike mammals where glutamate is excitatory (AMPA/NMDA).
pub fn build_receptor_types_from_csr_flywire(
    row_ptr: &[i32],
    n_synapses: usize,
    nt_map: &[f32],
) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = if row_ptr.len() > 1 {
        row_ptr.len() - 1
    } else {
        0
    };
    let mut syn_idx = 0usize;
    for pre in 0..n {
        let nt = nt_map.get(pre).copied().unwrap_or(1.0);
        // Drosophila-specific mapping:
        // +1 = ACh → nAChR (4, fast exc)
        // -1 = GABA → GABA_A (2, fast inh)
        // +2 = GLUT → GluCl (5, fast inh — Cl⁻ channel!)
        // -2 = DA/SER/OCT → mAChR (6, slow mod)
        let rec = if nt > 1.5 {
            5u8
        }
        // +2.0 → GluCl (GLUT, inhibitory!)
        else if nt > 0.5 {
            4u8
        }
        // +1.0 → nAChR (ACh, excitatory)
        else if nt < -1.5 {
            6u8
        }
        // -2.0 → mAChR (DA/SER/OCT, slow mod)
        else {
            2u8
        }; // -1.0 or 0 → GABA_A (GABA, fast inh)
        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(pre + 1).copied().unwrap_or(0) as usize).min(n_synapses);
        for _ in start..end {
            if syn_idx < receptors.len() {
                receptors[syn_idx] = rec;
            }
            syn_idx += 1;
        }
    }
    receptors
}

/// Build per-synapse receptor_types from textual FlyWire neuron NT labels.
///
/// Parses `neurons.csv.gz`-style labels and maps to Drosophila receptor types:
///
/// | Label | NT          | Receptor | Code | Effect |
/// |-------|-------------|----------|------|--------|
/// | ACh   | Acetylcholine | nAChR | 4 | Fast exc |
/// | GABA  | GABA        | GABA_A | 2 | Fast inh |
/// | GLUT  | Glutamate   | GluCl | 5 | **Inh** (Cl⁻) |
/// | DA    | Dopamine    | mAChR | 6 | Slow mod |
/// | SER   | Serotonin   | mAChR | 6 | Slow mod |
/// | OCT   | Octopamine  | mAChR | 6 | Slow mod |
/// | other | Unknown     | AMPA | 0 | Exc fallback |
///
/// `nt_labels[i]` = one of "ACh", "GABA", "GLUT", "DA", "SER", "OCT", or empty.
pub fn build_receptor_types_from_csv(
    row_ptr: &[i32],
    n_synapses: usize,
    nt_labels: &[String],
) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = if row_ptr.len() > 1 {
        row_ptr.len() - 1
    } else {
        0
    };
    let mut syn_idx = 0usize;
    for pre in 0..n {
        let rec = if let Some(label) = nt_labels.get(pre) {
            match label.as_str() {
                "ACh" => 4u8,                       // nAChR (fast exc)
                "GABA" => 2u8,                      // GABA_A (fast inh)
                "GLUT" => 5u8,                      // GluCl (fast inh — Cl⁻!)
                "DA" | "DOPAMINE" => 6u8,           // mAChR (slow mod)
                "SER" | "SEROTONIN" | "5HT" => 6u8, // mAChR
                "OCT" | "OCTOPAMINE" => 6u8,        // mAChR
                _ => 0u8,                           // AMPA fallback
            }
        } else {
            0u8 // AMPA fallback
        };
        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(pre + 1).copied().unwrap_or(0) as usize).min(n_synapses);
        for _ in start..end {
            if syn_idx < receptors.len() {
                receptors[syn_idx] = rec;
            }
            syn_idx += 1;
        }
    }
    receptors
}

// ═══════════════════════════════════════════════════════════
//  BrainIO
// ═══════════════════════════════════════════════════════════

impl BrainIO {
    /// Apply input signal to `external_I` buffer using per-channel encoding.
    ///
    /// `signal` length must equal `input_channels.len()`.
    ///
    /// | Encoding  | Behaviour |
    /// |-----------|-----------|
    /// | `Direct`  | `I[neuron] += signal[i] * scale` |
    /// | `Poisson` | Same as Direct (caller should generate Poisson spikes externally) |
    pub fn apply_input(&self, signal: &[f32], external_I: &mut [f32]) {
        for (ch, &val) in self.input_channels.iter().zip(signal) {
            let current = val * ch.scale;
            match ch.encoding {
                InputEncoding::Direct => {
                    for &idx in &ch.neuron_indices {
                        if idx < external_I.len() {
                            external_I[idx] += current;
                        }
                    }
                }
                InputEncoding::Poisson => {
                    for &idx in &ch.neuron_indices {
                        if idx < external_I.len() {
                            external_I[idx] += current;
                        }
                    }
                }
            }
        }
    }

    /// Reset `external_I` to just the bias (zero all, then add bias to all neurons).
    pub fn reset_current(&self, external_I: &mut [f32]) {
        external_I.fill(self.bias);
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_receptor_types_basic() {
        // 3 neurons: 0→1 (exc), 1→2 (inh), 2→0 (exc)
        let row_ptr = vec![0i32, 1, 2, 3];
        let n_syn = 3;
        let nt_types = vec![1.0, -1.0, 1.0];

        let rec = build_receptor_types_from_csr(&row_ptr, n_syn, &nt_types);
        assert_eq!(rec.len(), 3);
        // Neuron 0 (exc=1.0→AMPA=0), Neuron 1 (inh=-1.0→GABA_A=2), Neuron 2 (exc=1.0→AMPA=0)
        assert_eq!(rec[0], 0);
        assert_eq!(rec[1], 2);
        assert_eq!(rec[2], 0);
    }

    #[test]
    fn test_build_receptor_types_flywire() {
        // 4 neurons with different NT types — Drosophila mapping
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let n_syn = 4;
        let nt_map = vec![1.0, -1.0, 2.0, -2.0]; // ACh, GABA, GLUT, DA

        let rec = build_receptor_types_from_csr_flywire(&row_ptr, n_syn, &nt_map);
        assert_eq!(rec.len(), 4);
        assert_eq!(rec[0], 4); // +1.0 (ACh) → nAChR (4)
        assert_eq!(rec[1], 2); // -1.0 (GABA) → GABA_A (2)
        assert_eq!(rec[2], 5); // +2.0 (GLUT) → GluCl (5, inhibitory!)
        assert_eq!(rec[3], 6); // -2.0 (DA) → mAChR (6)
    }

    #[test]
    fn test_build_receptor_types_from_csv() {
        // 5 neurons with textual NT labels — Drosophila mapping
        let row_ptr = vec![0i32, 1, 2, 3, 4, 5];
        let n_syn = 5;
        let nt_labels = vec![
            "ACh".to_string(),
            "GABA".to_string(),
            "GLUT".to_string(),
            "DA".to_string(),
            "SER".to_string(),
        ];

        let rec = build_receptor_types_from_csv(&row_ptr, n_syn, &nt_labels);
        assert_eq!(rec.len(), 5);
        assert_eq!(rec[0], 4); // ACh → nAChR (fast exc)
        assert_eq!(rec[1], 2); // GABA → GABA_A (fast inh)
        assert_eq!(rec[2], 5); // GLUT → GluCl (fast inh!)
        assert_eq!(rec[3], 6); // DA → mAChR (slow mod)
        assert_eq!(rec[4], 6); // SER → mAChR (slow mod)
    }

    #[test]
    fn test_build_receptor_types_connectome_method() {
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2, 3],
            vec![1i32, 2, 0],
            vec![0.5f32, 1.0, 0.8],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        conn.build_receptor_types();
        assert_eq!(conn.receptor_types.len(), 3);
        assert_eq!(conn.receptor_types[0], 0); // AMPA
        assert_eq!(conn.receptor_types[1], 2); // GABA_A

        // Second call should be no-op
        let len_before = conn.receptor_types.len();
        conn.build_receptor_types();
        assert_eq!(conn.receptor_types.len(), len_before);
    }

    #[test]
    fn test_build_receptor_types_flywire_connectome_method() {
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2, 3, 4],
            vec![1i32, 2, 0, 3],
            vec![0.5f32, 1.0, 0.8, 0.3],
            vec![1.0, -1.0, 1.0, 1.0], // default nt_types
            vec![],
        );
        let nt_map = vec![1.0, -1.0, 2.0, -2.0]; // ACh, GABA, GLUT, DA
        conn.build_receptor_types_flywire(&nt_map);
        assert_eq!(conn.receptor_types.len(), 4);
        assert_eq!(conn.receptor_types[0], 4); // +1.0 (ACh) → nAChR
        assert_eq!(conn.receptor_types[1], 2); // -1.0 (GABA) → GABA_A
        assert_eq!(conn.receptor_types[2], 5); // +2.0 (GLUT) → GluCl
    }

    #[test]
    fn test_save_load_receptor_types() {
        let mut conn = ConnectomeData::new(
            vec![0i32, 2, 4, 6],
            vec![1i32, 2, 0, 1, 2, 0],
            vec![0.5f32, 1.0, 0.8, 0.3, 0.2, 0.7],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        conn.build_receptor_types();

        let mut buf = Vec::new();
        conn.save_baked(&mut buf).unwrap();

        let loaded = ConnectomeData::load_baked(&mut &buf[..]).unwrap();
        assert_eq!(loaded.receptor_types, conn.receptor_types);
        assert_eq!(loaded.n_neurons, conn.n_neurons);
        assert_eq!(loaded.n_synapses, conn.n_synapses);
        assert_eq!(loaded.values, conn.values);
    }

    #[test]
    fn test_save_load_empty_receptor_types() {
        let conn = ConnectomeData::new(
            vec![0i32, 2, 4, 6],
            vec![1i32, 2, 0, 1, 2, 0],
            vec![0.5f32; 6],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        // No receptor_types built — should still save/load fine
        let mut buf = Vec::new();
        conn.save_baked(&mut buf).unwrap();
        let loaded = ConnectomeData::load_baked(&mut &buf[..]).unwrap();
        assert!(loaded.receptor_types.is_empty());
    }
}
