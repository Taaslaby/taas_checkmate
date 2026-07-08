//! Universal connectome data container.
//!
//! [`ConnectomeData`] holds the CSR (Compressed Sparse Row) representation
//! of any directed graph with signed weights (excitatory/inhibitory).
//!
//! Use it to build a connectome from any source (FlyWire, MANC, synthetic, etc.)
//! and then pass it to [`BrainSimulator`](crate::sim::BrainSimulator) for simulation.

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
        let n_neurons = if row_ptr.len() > 1 { row_ptr.len() - 1 } else { 0 };
        let n_synapses = values.len();

        assert_eq!(row_ptr.len(), n_neurons + 1,
            "row_ptr must have n_neurons + 1 elements");
        assert_eq!(col_indices.len(), n_synapses,
            "col_indices length must match values length");
        assert!(nt_types.is_empty() || nt_types.len() >= n_neurons,
            "nt_types must have at least n_neurons elements");

        Self {
            n_neurons,
            n_synapses,
            row_ptr,
            col_indices,
            values,
            nt_types: if nt_types.is_empty() { vec![1.0; n_neurons] } else { nt_types },
            groups: groups.into_iter()
                .map(|(name, indices)| NeuronGroup { name: name.to_string(), indices })
                .collect(),
        }
    }

    /// Get a named group by name. Returns None if not found.
    pub fn group(&self, name: &str) -> Option<&[usize]> {
        self.groups.iter()
            .find(|g| g.name == name)
            .map(|g| g.indices.as_slice())
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
            &self.row_ptr, &self.col_indices, &self.values,
            &self.nt_types, &perm,
        );

        self.row_ptr = new_rp;
        self.col_indices = new_ci;
        self.values = new_vals;
        self.nt_types = new_nt;
        self.groups = crate::rcm::apply_permutation_to_groups(&self.groups, &perm);

        perm
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
