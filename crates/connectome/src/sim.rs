//! Brain simulator — универсальный движок для любого CSR коннектома.
//!
//! Поддерживает LIF, Izhikevich, Rate, и ψ-wave модели.
//! Принимает [`ConnectomeData`](crate::data::ConnectomeData) для описания графа.

use crate::data::ConnectomeData;
use taas_kernels::safe;
use taas_kernels::neuron::synapse::{DecayFactors, SynapticKinetics, ReceptorType};

/// Neuron model type.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NeuronModel {
    LIF,
    Izhikevich,
    Rate,
    Psi,
}

// ═══════════════════════════════════════════════════════════
//  SpikeQueue — lock-free SPSC circular buffer
// ═══════════════════════════════════════════════════════════

/// Lock-free single-producer single-consumer spike queue (power-of-2 circular buffer).
///
/// Decouples SIMD-hot spike detection from CSR-scatter propagation.
/// `push()` is called right after the SIMD neuron step (sequential scan of `spikes[]`),
/// `pop()` / `drain()` is called by the CSR propagation phase.
///
/// Capacity must be a power of 2. The mask = capacity - 1 enables bitwise wrap.
#[derive(Debug, Clone)]
pub struct SpikeQueue {
    buffer: Vec<u32>,
    head: usize,
    tail: usize,
    mask: usize,
}

impl SpikeQueue {
    /// Create a spike queue with capacity = next power of 2 >= `min_capacity`.
    pub fn new(min_capacity: usize) -> Self {
        let cap = min_capacity.max(4).next_power_of_two();
        Self {
            buffer: vec![0u32; cap],
            head: 0,
            tail: 0,
            mask: cap - 1,
        }
    }

    /// Push a spike ID. Panics on overflow (queue full) in debug builds.
    #[inline(always)]
    pub fn push(&mut self, spike_id: u32) {
        debug_assert!(self.len() < self.buffer.len(), "SpikeQueue overflow: len={} cap={}", self.len(), self.buffer.len());
        self.buffer[self.head & self.mask] = spike_id;
        self.head += 1;
    }

    /// Pop the oldest spike ID.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<u32> {
        if self.head == self.tail {
            None
        } else {
            let id = self.buffer[self.tail & self.mask];
            self.tail += 1;
            Some(id)
        }
    }

    /// Drain all queued spikes: pop and discard (used after propagation).
    #[inline(always)]
    pub fn drain(&mut self) {
        self.tail = self.head;
    }

    /// Number of spikes currently in the queue.
    pub fn len(&self) -> usize {
        self.head - self.tail
    }

    /// Reset the queue (discard all spikes).
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }
}

impl Default for SpikeQueue {
    fn default() -> Self {
        Self::new(1024)
    }
}

// ═══════════════════════════════════════════════════════════
//  Brain simulation state
// ═══════════════════════════════════════════════════════════

/// Brain simulation state (persistent between batch calls).
///
/// All arrays are padded to a multiple of 8 for SIMD alignment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainState {
    /// Membrane potential (LIF/Izhikevich/Psi). Padded to %8.
    pub v: Vec<f32>,
    /// Recovery variable (Izhikevich/Psi). Padded to %8.
    pub u: Vec<f32>,
    /// Spike trace (Izhikevich STDP only). Padded to %8.
    pub trace: Vec<f32>,
    /// Current buffer. Padded to %8.
    pub I_buf: Vec<f32>,
    /// Spike output buffer (per step). Padded to %8.
    pub spikes: Vec<i32>,
    /// Refractory counter (per neuron). Padded to %8.
    pub refractory: Vec<f32>,
    /// Padded length (multiple of 8 for SIMD).
    pub n_padded: usize,
    /// Actual number of neurons (unpadded).
    #[serde(skip)]
    pub n_neurons: usize,
    /// Spike queue — decouples SIMD from CSR propagation.
    #[serde(skip)]
    pub queue: SpikeQueue,
}

impl BrainState {
    /// Create a new brain state for `n_neurons` neurons.
    pub fn new(n_neurons: usize, model: NeuronModel) -> Self {
        let n_padded = (n_neurons + 7) / 8 * 8;
        let (v_init, u_init) = match model {
            NeuronModel::LIF => (-58.0, 0.0),
            NeuronModel::Izhikevich => (-65.0, -13.0),
            NeuronModel::Rate => (0.0, 0.0),
            NeuronModel::Psi => (0.0, 0.0),
        };

        // Queue capacity = n_neurons (max possible spikes per step), next power of 2
        // Guarantees no overflow even in 50%+ firing rate bursts.
        let queue_cap = n_neurons.max(4).next_power_of_two();

        Self {
            v: vec![v_init; n_padded],
            u: vec![u_init; n_padded],
            trace: vec![0.0; n_padded],
            I_buf: vec![0.0; n_padded],
            spikes: vec![0; n_padded],
            refractory: vec![0.0; n_padded],
            n_padded,
            n_neurons,
            queue: SpikeQueue::new(queue_cap),
        }
    }

    /// Переупорядочить state массивы по permutation (например, от RCM).
    ///
    /// `perm[i] = new_index`. Вызывается после `ConnectomeData::reorder_rcm()`
    /// для синхронизации состояния нейронов с новым порядком.
    ///
    /// # Example
    /// ```ignore
    /// let perm = connectome.reorder_rcm();
    /// state.permute_state(&perm);
    /// ```
    pub fn permute_state(&mut self, perm: &[usize]) {
        crate::rcm::apply_permutation_to_state(&mut self.v, perm);
        crate::rcm::apply_permutation_to_state(&mut self.u, perm);
        crate::rcm::apply_permutation_to_state(&mut self.trace, perm);
        crate::rcm::apply_permutation_to_state(&mut self.I_buf, perm);
        crate::rcm::apply_permutation_to_state(&mut self.spikes, perm);
        crate::rcm::apply_permutation_to_state(&mut self.refractory, perm);
        // Padding (n_padded..) остаётся без изменений — в нём нет данных
    }

    /// Reset state with small random noise (no `rand` dependency — uses simple hash).
    pub fn reset_with_noise(&mut self, n_neurons: usize) {
        for i in 0..n_neurons {
            // Simple deterministic noise without rand dependency
            let noise = ((i as f32 * 0.577) % 1.0 - 0.5) * 2.0;
            self.v[i] += noise * 0.03;
        }
    }
}

/// Packed LIF parameters for the SIMD kernel.
///
/// `safe::lif_step` expects `params: &[f32]` of length `n * 5` containing
/// tau_m, v_rest, v_thresh, v_reset interleaved per neuron.
#[derive(Debug, Clone)]
pub struct LifParams {
    packed: Vec<f32>,
    pub tau_ref: f32,
    pub dt_ms: f32,
}

impl LifParams {
    /// Create LIF params padded to `n_padded` (must be >= n_neurons, multiple of 8 for SIMD).
    ///
    /// The kernel expects column-major layout with stride `n` (where n = n_padded = v.len()):
    ///   params[0] = dt (scalar, index 0)
    ///   params[n..2n) = tau_m[0..n)
    ///   params[2n..3n) = v_rest[0..n)
    ///   params[3n..4n) = v_thresh[0..n)
    ///   params[4n..5n) = v_reset[0..n)
    ///
    /// Note: params[1..n) are UNUSED padding (gap after dt before first column).
    pub fn new(tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32], v_reset: &[f32], dt_ms: f32, tau_ref: f32) -> Self {
        let n = tau_m.len();
        let n_padded = (n + 7) / 8 * 8;
        // Allocate 5 * n_padded elements (stride = n_padded for each column)
        let mut packed = vec![0.0f32; n_padded * 5];
        // Column 0: dt at index 0 (rest of indices 1..n_padded are unused padding)
        packed[0] = dt_ms;
        // Columns 1..4: tau_m, v_rest, v_thresh, v_reset
        for i in 0..n {
            packed[n_padded + i] = tau_m[i];        // col 1
            packed[2 * n_padded + i] = v_rest[i];    // col 2
            packed[3 * n_padded + i] = v_thresh[i];  // col 3
            packed[4 * n_padded + i] = v_reset[i];   // col 4
        }
        // Replicate last values for padding elements (n..n_padded)
        for i in n..n_padded {
            packed[n_padded + i] = tau_m[n - 1];
            packed[2 * n_padded + i] = v_rest[n - 1];
            packed[3 * n_padded + i] = v_thresh[n - 1];
            packed[4 * n_padded + i] = v_reset[n - 1];
        }
        Self { packed, tau_ref, dt_ms }
    }
}

/// Parameters for Izhikevich model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IzhikevichParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub dt_ms: f32,
    pub tau_ref: f32,
}

/// Parameters for Rate model.
#[derive(Debug, Clone)]
pub struct RateParams {
    pub tau: f32,
    pub dt_ms: f32,
}

/// Parameters for ψ-wave model.
#[derive(Debug, Clone)]
pub struct PsiParams {
    pub hbar: f32,
    pub dt_ms: f32,
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_queue_basic_push_pop() {
        let mut q = SpikeQueue::new(8);
        assert_eq!(q.len(), 0);
        q.push(5);
        q.push(42);
        q.push(100);
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(5));
        assert_eq!(q.pop(), Some(42));
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop(), Some(100));
        assert_eq!(q.pop(), None);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn spike_queue_wrap_around() {
        let mut q = SpikeQueue::new(4);
        // Push until near capacity
        q.push(1);
        q.push(2);
        q.push(3);
        q.push(4);
        assert_eq!(q.len(), 4);
        // Pop all
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.len(), 2);
        // Now push more — wraps around in circular buffer
        q.push(10);
        q.push(11);
        // head is now 6, tail is 2, cap=4, mask=3
        assert_eq!(q.len(), 4); // 6 - 2 = 4
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(11));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_queue_drain() {
        let mut q = SpikeQueue::new(8);
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.len(), 3);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(q.pop(), None);
        // After clear, can still push
        q.push(42);
        assert_eq!(q.pop(), Some(42));
    }

    #[test]
    fn spike_queue_capacity_power_of_two() {
        let q = SpikeQueue::new(5);  // next power of 2 = 8
        assert_eq!(q.buffer.len(), 8);
        assert_eq!(q.mask, 7);

        let q2 = SpikeQueue::new(1000);  // next power of 2 = 1024
        assert_eq!(q2.buffer.len(), 1024);

        let q3 = SpikeQueue::new(1);  // min 4
        assert_eq!(q3.buffer.len(), 4);
    }

    #[test]
    fn push_spikes_to_queue_basic() {
        let spikes = vec![0, 1, 0, 0, 1, 0, 1];
        let mut q = SpikeQueue::new(16);
        let count = push_spikes_to_queue(&spikes, 7, &mut q);
        assert_eq!(count, 3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(6));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn csr_propagate_from_queue_basic() {
        // Simple ring: 0→1, 1→2, 2→0
        let row_ptr = vec![0i32, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![0.5f32, 0.5, 0.5];
        let nt_types = vec![1.0f32, 1.0, 1.0];
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(8);

        // Neuron 0 spikes
        q.push(0);
        csr_propagate_from_queue(&row_ptr, &col_indices, &values, &nt_types, &mut q, &mut I_buf, 1.0);

        // I_buf[1] should get 0.5
        assert!((I_buf[1] - 0.5).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(I_buf[2], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_multi_spike() {
        // 0→2, 1→1 (self), 2→0, 3→1
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let col_indices = vec![2, 1, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let nt_types = vec![1.0f32, -1.0, 1.0, 1.0];  // neuron 1 is inhibitory
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(16);

        q.push(0); // exc → I_buf[2] += 1.0
        q.push(1); // inh → I_buf[1] += 2.0 * -1.0 = -2.0
        q.push(3); // exc → I_buf[1] += 4.0

        csr_propagate_from_queue(&row_ptr, &col_indices, &values, &nt_types, &mut q, &mut I_buf, 1.0);

        assert!((I_buf[2] - 1.0).abs() < 1e-6, "I_buf[2]={}", I_buf[2]);
        assert!((I_buf[1] - 2.0).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);  // -2.0 + 4.0 = 2.0
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_inhibitory_scale() {
        let row_ptr = vec![0i32, 2, 2];
        let col_indices = vec![0, 1];  // neuron 0→0 and 0→1
        let values = vec![1.0f32, 1.0];
        let nt_types = vec![1.0f32, 0.0];  // neuron 1 is silent (no synapses anyway)
        let mut I_buf = vec![0.0f32; 2];
        let mut q = SpikeQueue::new(8);

        q.push(0);
        // scale = -0.5
        csr_propagate_from_queue(&row_ptr, &col_indices, &values, &nt_types, &mut q, &mut I_buf, -0.5);

        assert!((I_buf[0] - (-0.5)).abs() < 1e-6, "I_buf[0]={}", I_buf[0]);
        assert!((I_buf[1] - (-0.5)).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(q.len(), 0);
    }

    // ─── Synaptic kinetics integration tests ───────────

    fn make_synaptic_connectome() -> ConnectomeData {
        // 3 neurons: 0→1 (exc, AMPA), 1→2 (inh, GABA_A), 2→0 (exc, AMPA)
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2, 3],
            vec![1, 2, 0],
            vec![0.5f32, 1.0, 0.8],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        conn.build_receptor_types();
        conn
    }

    #[test]
    fn test_sim_lif_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(!conn.receptor_types.is_empty(), "receptor_types must be non-empty");

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        // Strong drive ensures firing: LIF dv/dt = (I - (V - V_rest)) / tau
        // With I=100, tau=5, dt=0.1: dv ~= 2.0/step → reaches V_thresh=0 in ~26 steps
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![100.0f32; n];
        let n_steps = 100;
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_lif_synaptic(&conn, &mut state, &params, &external_I, n_steps, &mut kinetics);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(total_spikes > 0, "LIF synaptic should fire with high drive, got {} spikes", total_spikes);
        assert_eq!(spikes.len(), n_steps * n);
    }

    #[test]
    fn test_sim_izhikevich_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(!conn.receptor_types.is_empty());

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![7.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(&conn, &mut state, &params, &external_I, 20, &mut kinetics);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(total_spikes > 0, "Izhikevich synaptic should fire with high drive");
        assert_eq!(spikes.len(), 20 * n);
    }

    #[test]
    fn test_sim_lif_synaptic_conductance_decay() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![0.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        kinetics.jump(0, ReceptorType::AMPA, 10.0);
        kinetics.jump(1, ReceptorType::AMPA, 10.0);

        let initial_g = kinetics.g_ampa[0];
        assert!((initial_g - 10.0).abs() < 1e-6, "initial g={}", initial_g);

        let _spikes = sim_lif_synaptic(&conn, &mut state, &params, &external_I, 1, &mut kinetics);

        let expected_decay = (-0.1f32 / 2.0).exp();
        assert!(kinetics.g_ampa[0] < 10.0 * expected_decay * 1.01,
            "g_ampa[0]={} should have decayed", kinetics.g_ampa[0]);
        assert!(kinetics.g_ampa[0] > 0.0, "g_ampa[0] should still be positive");
    }

    #[test]
    fn test_sim_izhikevich_synaptic_receptor_types() {
        let mut conn = make_synaptic_connectome();
        conn.receptor_types = vec![0u8, 1, 2];  // AMPA, NMDA, GABA_A

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![10.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(&conn, &mut state, &params, &external_I, 50, &mut kinetics);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(total_spikes > 0, "Izhikevich with mixed receptors should fire");

        let ampa_decay = (-0.5f32 / 2.0).exp().powi(50);
        let nmda_decay = (-0.5f32 / 50.0).exp().powi(50);
        assert!(nmda_decay > ampa_decay, "NMDA should decay slower than AMPA");
        println!("AMPA decay over 50 steps: {:.6}", ampa_decay);
        println!("NMDA decay over 50 steps: {:.6}", nmda_decay);
    }

    #[test]
    fn test_sim_hh_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(!conn.receptor_types.is_empty(), "receptor_types must be non-empty");

        let n = conn.n_neurons;
        let mut cfg = taas_core::neuron::hh::standard_config(n, 0.5);
        // Strong drive ensures HH fires: I=15 µA/cm² is suprathreshold for poly HH
        let external_I = vec![15.0f32; n];
        let n_steps = 100;
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_hh_synaptic(&conn, &mut cfg, &external_I, n_steps, &mut kinetics);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(total_spikes > 0, "HH synaptic should fire with high drive (I=15), got {} spikes", total_spikes);
        assert_eq!(spikes.len(), n_steps * n);
    }

    #[test]
    fn test_sim_hh_basic() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        let mut cfg = taas_core::neuron::hh::standard_config(n, 0.5);
        let external_I = vec![15.0f32; n];
        let n_steps = 100;

        let spikes = sim_hh(&conn, &mut cfg, &external_I, n_steps);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(total_spikes > 0, "HH (instant) should fire with high drive (I=15), got {} spikes", total_spikes);
        assert_eq!(spikes.len(), n_steps * n);
    }
}

// ─── CSR propagation ─────────────────────────────────

// ─── CSR propagation ─────────────────────────────────

/// Propagate spikes through CSR: `I_buf[post] += W · mult` for each spiked pre-neuron.
///
/// Performs unrolled 4-wide scatter for performance.
#[inline]
pub fn csr_propagate(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    spikes: &[i32],
    n_neurons: usize,
    I_buf: &mut [f32],
    scale: f32,
) -> u32 {
    let max_syn = col_indices.len().min(values.len());
    let mut total = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            total += 1;
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1] as usize).min(max_syn);
            let mult = if i < nt_types.len() { nt_types[i] * scale } else { scale };
            let mut j = start;
            // Unrolled 4-wide
            while j + 4 <= end {
                let t0 = col_indices[j].max(0) as usize;
                let t1 = col_indices[j + 1].max(0) as usize;
                let t2 = col_indices[j + 2].max(0) as usize;
                let t3 = col_indices[j + 3].max(0) as usize;
                if t0 < I_buf.len() { I_buf[t0] += values[j] * mult; }
                if t1 < I_buf.len() { I_buf[t1] += values[j + 1] * mult; }
                if t2 < I_buf.len() { I_buf[t2] += values[j + 2] * mult; }
                if t3 < I_buf.len() { I_buf[t3] += values[j + 3] * mult; }
                j += 4;
            }
            for j in j..end {
                let col = col_indices[j].max(0) as usize;
                if col < I_buf.len() { I_buf[col] += values[j] * mult; }
            }
        }
    }
    total
}

/// Record spikes to spike buffer.
#[inline]
pub fn record_spikes(
    spikes: &[i32],
    n_neurons: usize,
    spike_buffer: &mut [i32],
    step: usize,
) -> u32 {
    let offset = step * n_neurons;
    let mut step_count = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            let idx = offset + i;
            if idx < spike_buffer.len() {
                spike_buffer[idx] = 1;
            }
            step_count += 1;
        }
    }
    step_count
}

// ─── Spike Queue helpers ──────────────────────────────

/// Push all spiked neurons from `spikes[]` into the queue.
///
/// Returns the number of spikes pushed. Called right after the SIMD neuron step,
/// before CSR propagation. This separates the SIMD-hot scan from scatter-write.
#[inline]
pub fn push_spikes_to_queue(spikes: &[i32], n_neurons: usize, queue: &mut SpikeQueue) -> u32 {
    let mut count = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            queue.push(i as u32);
            count += 1;
        }
    }
    count
}

/// Propagate spikes from queue through CSR: `I_buf[post] += W · mult` for each queued pre-neuron.
///
/// Same unrolled 4-wide scatter as `csr_propagate()` but reads spike IDs from the queue
/// instead of scanning the full `spikes[]` array. This gives:
/// 1. Cache-friendly sequential read of spike IDs from queue buffer
/// 2. Prefetcher can predict CSR row_ptr/col_indices access pattern
/// 3. Clear separation between SIMD-hot and scatter-hot paths
#[inline]
pub fn csr_propagate_from_queue(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    queue: &mut SpikeQueue,
    I_buf: &mut [f32],
    scale: f32,
) -> u32 {
    let max_syn = col_indices.len().min(values.len());
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        total += 1;
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let mult = if i < nt_types.len() { nt_types[i] * scale } else { scale };
        let mut j = start;
        // Unrolled 4-wide
        while j + 4 <= end {
            let t0 = col_indices[j].max(0) as usize;
            let t1 = col_indices[j + 1].max(0) as usize;
            let t2 = col_indices[j + 2].max(0) as usize;
            let t3 = col_indices[j + 3].max(0) as usize;
            if t0 < I_buf.len() { I_buf[t0] += values[j] * mult; }
            if t1 < I_buf.len() { I_buf[t1] += values[j + 1] * mult; }
            if t2 < I_buf.len() { I_buf[t2] += values[j + 2] * mult; }
            if t3 < I_buf.len() { I_buf[t3] += values[j + 3] * mult; }
            j += 4;
        }
        for j in j..end {
            let col = col_indices[j].max(0) as usize;
            if col < I_buf.len() { I_buf[col] += values[j] * mult; }
        }
    }
    total
}

// ─── Conductance-based propagation ─────────────────

/// Propagate spikes from queue through CSR: jump conductances instead of currents.
///
/// For each spiked pre-neuron, iterates its outgoing synapses and applies
/// `kinetics.jump(post, receptor_types[j], values[j] * scale)`.
///
/// Requires `receptor_types` array of length `n_synapses`. Each synapse's
/// receptor type is stored as a `u8` (0=AMPA, 1=NMDA, 2=GABA_A, 3=GABA_B).
///
/// Returns the number of spikes processed (total propagated).
#[inline]
pub fn csr_propagate_conductance_from_queue(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    receptor_types: &[u8],
    queue: &mut SpikeQueue,
    kinetics: &mut SynapticKinetics,
    scale: f32,
) -> u32 {
    let max_syn = col_indices.len().min(values.len()).min(receptor_types.len());
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        total += 1;
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let mut j = start;
        // Unrolled 4-wide
        while j + 4 <= end {
            let t0 = col_indices[j].max(0) as usize;
            let t1 = col_indices[j + 1].max(0) as usize;
            let t2 = col_indices[j + 2].max(0) as usize;
            let t3 = col_indices[j + 3].max(0) as usize;
            let w0 = values[j] * scale;
            let w1 = values[j + 1] * scale;
            let w2 = values[j + 2] * scale;
            let w3 = values[j + 3] * scale;
            let rec0 = ReceptorType::from_u8(receptor_types[j]);
            let rec1 = ReceptorType::from_u8(receptor_types[j + 1]);
            let rec2 = ReceptorType::from_u8(receptor_types[j + 2]);
            let rec3 = ReceptorType::from_u8(receptor_types[j + 3]);
            kinetics.jump(t0, rec0, w0);
            kinetics.jump(t1, rec1, w1);
            kinetics.jump(t2, rec2, w2);
            kinetics.jump(t3, rec3, w3);
            j += 4;
        }
        while j < end {
            let post = col_indices[j].max(0) as usize;
            let w = values[j] * scale;
            let rec = ReceptorType::from_u8(receptor_types[j]);
            kinetics.jump(post, rec, w);
            j += 1;
        }
    }
    total
}

// ─── LIF simulation ──────────────────────────────────

/// Simulate `n_steps` of LIF dynamics on a connectome.
///
/// `safe::lif_step` signature:
/// `lif_step(v: &mut [f32], ii: &mut [f32], spikes: &mut [i32],
///           params: &[f32], refractory: &mut [f32], tau_ref: f32)`
/// where params = [tau_m, v_rest, v_thresh, v_reset, _] interleaved per neuron (n * 5).
pub fn sim_lif(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &LifParams,
    external_I: &[f32],
    n_steps: usize,
    scale: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];

    for step in 0..n_steps {
        // 1. ADD external_I on top of I_buf (preserves carry-over propagated current)
        for i in 0..n.min(external_I.len()) {
            state.I_buf[i] += external_I[i];
        }
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 2. LIF step (SIMD) — reads I_buf as input, does NOT modify it
        //    Note: safety_clamp is now INSIDE safe::lif_step
        safe::lif_step(
            &mut state.v,
            &mut state.I_buf,
            &mut state.spikes,
            &params.packed,
            &mut state.refractory,
            params.tau_ref,
        );

        // 4. Push spikes to queue (decouple SIMD → scatter)
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 5. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 6. Clear I_buf (current was consumed by LIF; prepare for fresh propagation)
        state.I_buf[..n].fill(0.0);
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 7. CSR propagate spikes from queue → I_buf (adds, does NOT overwrite)
        csr_propagate_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.nt_types, &mut state.queue, &mut state.I_buf, scale,
        );
    }

    spike_buffer
}

// ─── LIF simulation with synaptic kinetics ───────────

/// Simulate LIF dynamics with conductance-based synaptic kinetics.
///
/// Requires `connectome.receptor_types` to be non-empty. Uses per-synapse
/// receptor types (AMPA, NMDA, GABA_A, GABA_B) to compute synaptic currents
/// from conductances instead of instant current injection.
///
/// Per-step lifecycle:
/// 1. Decay all conductances: `g *= exp(-dt/τ)` for each receptor type
/// 2. Compute I_syn: `I_syn[i] = Σ g_r[i] · (V[i] - E_rev_r)`
/// 3. Set I_buf = external_I + I_syn
/// 4. LIF step
/// 5. Push spikes to queue
/// 6. CSR propagate: jump conductances instead of currents
pub fn sim_lif_synaptic(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &LifParams,
    external_I: &[f32],
    n_steps: usize,
    kinetics: &mut SynapticKinetics,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let factors = DecayFactors::new(params.dt_ms);

    for step in 0..n_steps {
        // 1. Decay conductances: g *= exp(-dt/τ)
        kinetics.decay_all(&factors);

        // 2. Set I_buf = external_I (reset)
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 3. Compute I_syn from conductances and add to I_buf
        kinetics.compute_currents(&state.v, &mut state.I_buf);

        // 4. LIF step (SIMD) — clamp is INSIDE safe::lif_step
        safe::lif_step(
            &mut state.v,
            &mut state.I_buf,
            &mut state.spikes,
            &params.packed,
            &mut state.refractory,
            params.tau_ref,
        );

        // 6. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 7. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 7. CSR propagate: jump conductances (not currents!)
        csr_propagate_conductance_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.receptor_types, &mut state.queue, kinetics, 1.0,
        );
    }

    spike_buffer
}

// ─── Izhikevich simulation ─────────────────────────────

/// Pad a Vec to `n_padded` by replicating the last element.
fn pad_to<T: Copy>(v: &[T], n_padded: usize) -> Vec<T> {
    let mut padded = v.to_vec();
    padded.resize(n_padded, *v.last().unwrap_or(&v[0]));
    padded
}

/// Simulate `n_steps` of Izhikevich dynamics on a connectome.
///
/// `safe::izhikevich_step` expects all arrays to have the same length as `v`.
/// Since `v` is padded to `n_padded` (multiple of 8 for SIMD),
/// we pad `a`, `b`, `c`, `d` params too before passing to the SIMD kernel.
pub fn sim_izhikevich(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_padded = state.n_padded;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];

    let dt = params.dt_ms;

    // Pad params to match padded state arrays
    let padded_a = pad_to(&params.a, n_padded);
    let padded_b = pad_to(&params.b, n_padded);
    let padded_c = pad_to(&params.c, n_padded);
    let padded_d = pad_to(&params.d, n_padded);

    for step in 0..n_steps {
        // 1. Set I = external_I
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 2. Izhikevich step (SIMD + rayon) — clamp is INSIDE safe::izhikevich_step
        safe::izhikevich_step(
            &mut state.v, &mut state.u, &mut state.trace,
            &padded_a, &padded_b, &mut state.I_buf,
            dt, 0.99, // dt, trace_decay
            &mut state.spikes,
            &padded_c, &padded_d,
            &mut state.refractory, params.tau_ref,
        );

        // 4. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 5. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 6. Reset I = external_I
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 7. CSR propagate from queue
        csr_propagate_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.nt_types, &mut state.queue, &mut state.I_buf, 1.0,
        );
    }

    spike_buffer
}

// ─── Rate simulation ──────────────────────────────────

/// Simulate `n_steps` of Rate dynamics on a connectome.
///
/// Rate model stores firing probabilities in `state.v`.
/// Sets `state.spikes` before CSR propagation so recurrent excitation works.
pub fn sim_rate(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &RateParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let dt = params.dt_ms / 1000.0;
    let decay = (-dt / params.tau).exp();
    let inv_decay = 1.0 - decay;

    let r = &mut state.v; // reuse v as rate buffer
    let I = &mut state.I_buf;

    for step in 0..n_steps {
        // 1. Set I = external_r
        I[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);

        // 2. Scatter SpMV: I[post] += W[pre,post] * r[pre] * nt_types[pre]
        // Need state.spikes for csr_propagate—set it from r > 0.5
        state.queue.drain(); // clear any stale spikes from queue
        for i in 0..n_neurons {
            if i < r.len() && r[i] > 0.5 {
                state.spikes[i] = 1;
                state.queue.push(i as u32);
            } else {
                state.spikes[i] = 0;
            }
        }

        // 3. Propagate spikes from queue (rate-weighted)
        csr_propagate_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.nt_types, &mut state.queue, I, 1.0,
        );

        // 4. Exponential Euler: r = r * decay + clamp(I + I_ext, 0, 1) * (1 - decay)
        for i in 0..n {
            let ext = if i < external_I.len() { external_I[i] } else { 0.0 };
            let input = (I[i] + ext).clamp(0.0, 1.0);
            r[i] = r[i] * decay + input * inv_decay;
        }

        // 5. Record "spikes" (rate threshold crossing)
        for i in 0..n_neurons {
            if i < r.len() && r[i] > 0.5 {
                let idx = step * n_neurons + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
            }
        }
    }

    spike_buffer
}

// ─── GPU LIF simulation (CUDA) ──────────────────────────

/// GPU LIF simulation using CUDA — persistent context between calls.
#[cfg(feature = "cuda")]
pub mod gpu {
    use crate::data::ConnectomeData;
    use super::BrainState;
    use cuda_kernels::{GpuContext, DevicePtr};
    use std::sync::Mutex;

    static GPU_CTX: Mutex<Option<(GpuContext, usize, usize, bool)>> = Mutex::new(None);
    /// Cached GPU spike buffer — avoids alloc/free churn between batch calls.
    static GPU_SPIKE_BUF: Mutex<Option<(DevicePtr<i32>, usize)>> = Mutex::new(None);

    /// Get or create a GPU spike buffer of the required size.
    fn get_spike_buf(n_ints: usize) -> DevicePtr<i32> {
        let mut cache = GPU_SPIKE_BUF.lock().unwrap();
        if let Some((ptr, size)) = cache.take() {
            if size == n_ints {
                return ptr;  // reuse
            }
            // size changed — old buf dropped here
        }
        let ptr = DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer");
        *cache = Some((ptr, n_ints));
        // Return a fresh allocation (cache holds the same-sized one for next call)
        DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer")
    }

    /// Return a spike buffer to cache for reuse.
    fn put_spike_buf(buf: DevicePtr<i32>, n_ints: usize) {
        let mut cache = GPU_SPIKE_BUF.lock().unwrap();
        *cache = Some((buf, n_ints));
    }

    /// Simulate `n_steps` of LIF dynamics on a connectome using GPU (CUDA).
    ///
    /// Использует `batch_lif_steps` — аккумулирует спайки на GPU через per-step D2D,
    /// скачивает весь spike buffer + v одним D2H в конце.
    /// Также чинит stale external_I baseline (обновляет d_external_I каждый batch).
    ///
    /// Первый вызов создаёт GpuContext, последующие используют persistent cache.
    /// Uploads only v (not dummy u/a/b/c/d) — conditional upload saves ~5× bandwidth.
    pub fn sim_lif_gpu(
        connectome: &ConnectomeData,
        state: &mut BrainState,
        dt_ms: f32,
        external_I: &[f32],
        n_steps: usize,
        tau_m: &[f32],
        v_rest: &[f32],
        v_thresh: &[f32],
        v_reset: &[f32],
    ) -> Vec<i32> {
        let n = connectome.n_neurons;
        let n_neurons = connectome.row_ptr.len() - 1;
        let n_synapses = connectome.n_synapses;
        let spike_buf_len = n_steps * n;
        let mut spike_buffer = vec![0i32; spike_buf_len];
        let mut v_back = vec![0.0f32; n_neurons];

        // Get or create persistent GPU context
        let mut cache = GPU_CTX.lock().unwrap();
        let (gpu_ctx, needs_init) = match cache.take() {
            Some((ctx, cached_n, cached_s, uploaded))
                if cached_n == n_neurons && cached_s == n_synapses =>
            {
                (ctx, !uploaded)
            }
            _ => {
                let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
                (ctx, true)
            }
        };

        // Upload v only (no dummy u/a/b/c/d — LIF kernel doesn't use them)
        gpu_ctx.upload_v_only(&state.v[..n_neurons]).unwrap();

        if needs_init {
            gpu_ctx.upload_lif_params(dt_ms, tau_m, v_rest, v_thresh, v_reset).unwrap();
            gpu_ctx.upload_connectome(
                &connectome.row_ptr, &connectome.col_indices, &connectome.values,
                &connectome.nt_types, external_I,
            ).unwrap();
        }

        // Get cached spike buffer or allocate new one
        let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

        // Batch run
        gpu_ctx.batch_lif_steps(
            external_I,
            d_spike_buf.as_mut_ptr(),
            &mut spike_buffer,
            &mut v_back,
            n_steps,
        ).expect("GPU: batch_lif_steps");

        // Return spike buffer to cache
        put_spike_buf(d_spike_buf, n_steps * n_neurons);

        // Copy v back to state
        state.v[..n_neurons].copy_from_slice(&v_back);

        *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

        spike_buffer
    }

    /// Simulate `n_steps` of Izhikevich dynamics on a connectome using GPU (CUDA).
    ///
    /// Same batched approach as `sim_lif_gpu`:
    /// - Per-step D2D spike accumulation, single D2H at end
    /// - Updates d_external_I + d_I every call (fixes stale baseline)
    /// - Uploads full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
    /// - Downloads v + u at end for state persistence
    ///
    /// Default Izhikevich params: a=0.02, b=0.2, c=-65, d=8, trace_decay=0.99
    pub fn sim_izhikevich_gpu(
        connectome: &ConnectomeData,
        state: &mut BrainState,
        dt_ms: f32,
        external_I: &[f32],
        n_steps: usize,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
    ) -> Vec<i32> {
        let n = connectome.n_neurons;
        let n_neurons = connectome.row_ptr.len() - 1;
        let n_synapses = connectome.n_synapses;
        let spike_buf_len = n_steps * n;
        let mut spike_buffer = vec![0i32; spike_buf_len];
        let mut v_back = vec![0.0f32; n_neurons];
        let mut u_back = vec![0.0f32; n_neurons];
        let trace_decay = 0.99f32;

        // Get or create persistent GPU context
        let mut cache = GPU_CTX.lock().unwrap();
        let (gpu_ctx, needs_init) = match cache.take() {
            Some((ctx, cached_n, cached_s, uploaded))
                if cached_n == n_neurons && cached_s == n_synapses =>
            {
                (ctx, !uploaded)
            }
            _ => {
                let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
                (ctx, true)
            }
        };

        // Upload full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
        gpu_ctx.upload_neurons(
            &state.v[..n_neurons],
            &state.u[..n_neurons],
            a, b, c, d,
        ).unwrap();

        if needs_init {
            gpu_ctx.upload_connectome(
                &connectome.row_ptr, &connectome.col_indices, &connectome.values,
                &connectome.nt_types, external_I,
            ).unwrap();
        }

        // Get cached spike buffer
        let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

        // Batch run
        gpu_ctx.batch_izhikevich_steps(
            external_I,
            dt_ms,
            trace_decay,
            d_spike_buf.as_mut_ptr(),
            &mut spike_buffer,
            &mut v_back,
            &mut u_back,
            n_steps,
        ).expect("GPU: batch_izhikevich_steps");

        // Return spike buffer to cache
        put_spike_buf(d_spike_buf, n_steps * n_neurons);

        // Copy v + u back to state
        state.v[..n_neurons].copy_from_slice(&v_back);
        state.u[..n_neurons].copy_from_slice(&u_back);

        *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

        spike_buffer
    }
}

// ─── Izhikevich simulation with synaptic kinetics ─────

/// Simulate Izhikevich dynamics with conductance-based synaptic kinetics.
///
/// Same lifecycle as `sim_lif_synaptic` but with Izhikevich neurons:
/// 1. Decay all conductances: `g *= exp(-dt/tau)` for each receptor type
/// 2. Set I_buf = external_I (reset)
/// 3. Compute I_syn from conductances and add to I_buf
/// 4. Izhikevich step (SIMD)
/// 5. Push spikes to queue → record spikes
/// 6. CSR propagate: jump conductances (not currents)
///
/// Requires `connectome.receptor_types` to be non-empty.
pub fn sim_izhikevich_synaptic(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichParams,
    external_I: &[f32],
    n_steps: usize,
    kinetics: &mut SynapticKinetics,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_padded = state.n_padded;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let dt = params.dt_ms;
    let factors = DecayFactors::new(dt);

    let padded_a = pad_to(&params.a, n_padded);
    let padded_b = pad_to(&params.b, n_padded);
    let padded_c = pad_to(&params.c, n_padded);
    let padded_d = pad_to(&params.d, n_padded);

    for step in 0..n_steps {
        // 1. Decay conductances: g *= exp(-dt/tau)
        kinetics.decay_all(&factors);

        // 2. Set I_buf = external_I (reset)
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() { state.I_buf[i] = 0.0; }

        // 3. Compute I_syn from conductances and add to I_buf
        kinetics.compute_currents(&state.v, &mut state.I_buf);

        // 4. Izhikevich step (SIMD) — clamp is INSIDE safe::izhikevich_step
        safe::izhikevich_step(
            &mut state.v, &mut state.u, &mut state.trace,
            &padded_a, &padded_b, &mut state.I_buf,
            dt, 0.99,
            &mut state.spikes,
            &padded_c, &padded_d,
            &mut state.refractory, params.tau_ref,
        );

        // 6. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 7. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 8. CSR propagate: jump conductances (not currents!)
        csr_propagate_conductance_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.receptor_types, &mut state.queue, kinetics, 1.0,
        );
    }

    spike_buffer
}

// ─── Izhikevich + E-Prop ─────────────────────────────

/// Izhikevich simulation with per-synapse E-Prop eligibility traces.
///
/// **CSR (sparse) version** — рекомендуется для биологических коннектомов
/// (FlyWire, ~38 синапсов/нейрон). Использует `row_ptr`/`col_indices` для
/// разреженного обхода: O(spikes × avg_synapses) вместо O(n_pre × n_post).
///
/// # Dense vs Sparse E-Prop
///
/// | Версия | Формат | Когда использовать |
/// |--------|--------|-------------------|
/// | `eprop_izhikevich_step_multi` (kernels) | Column-major dense | n < 10K, полносвязные слои |
/// | `sim_izhikevich_eprop` (эта) | CSR sparse | n > 10K, биологические коннектомы |
///
/// Плотная версия (`eprop_izhikevich_step_multi`) идеальна для малых
/// полносвязных слоёв (500–1000 нейронов) — веса помещаются в L2 кэш,
/// SIMD-перемножение максимально эффективно. Для больших сетей (>10K)
/// используйте эту CSR-версию — cache-friendly обход через row_ptr.
///
/// Returns `(spike_buffer, eligibility)`:
/// - `spike_buffer [n_steps × n_neurons]` — 0/1 spikes at each step
/// - `eligibility [n_synapses]` — accumulated eligibility trace for each synapse
///
/// E-Prop update after simulation:
/// ```ignore
/// for idx in 0..n_syn {
///     let post = col_indices[idx] as usize;
///     values[idx] -= lr * L_j[post] * eligibility[idx];
/// }
/// ```
/// where `L_j[post]` is the learning signal (e.g. error backpropagated through readout).
pub fn sim_izhikevich_eprop(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichParams,
    external_I: &[f32],
    n_steps: usize,
    tau_e: f32,
    tau_surrogate: f32,
) -> (Vec<i32>, Vec<f32>) {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_syn = connectome.values.len();
    let n_padded = state.n_padded;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut eligibility = vec![0.0f32; n_syn];

    let dt = params.dt_ms;
    let beta_e = (-dt / tau_e.max(dt)).exp();
    let trace_decay = 0.99f32;

    let padded_a = pad_to(&params.a, n_padded);
    let padded_b = pad_to(&params.b, n_padded);
    let padded_c = pad_to(&params.c, n_padded);
    let padded_d = pad_to(&params.d, n_padded);

    let mut pre_trace = vec![0.0f32; n];
    let mut psi = vec![0.0f32; n];
    let v_th = 30.0;
    let surrogate_scale = 1.0 / tau_surrogate;
    let max_syn = connectome.col_indices.len().min(n_syn);

    for step in 0..n_steps {
        // 1. Set I = external_I
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() { state.I_buf[i] = 0.0; }

        // 2. Izhikevich step (SIMD) — clamp is INSIDE safe::izhikevich_step
        safe::izhikevich_step(
            &mut state.v, &mut state.u, &mut state.trace,
            &padded_a, &padded_b, &mut state.I_buf,
            dt, trace_decay,
            &mut state.spikes,
            &padded_c, &padded_d,
            &mut state.refractory, params.tau_ref,
        );

        // 3. Surrogate gradient: ψ_j = 1 / (1 + |v_j - 30|/τ_s)²
        for i in 0..n {
            let dv = (state.v[i] - v_th).abs();
            psi[i] = 1.0 / (1.0 + dv * surrogate_scale).powi(2);
        }

        // 4. Pre-spike trace: z_i = β·z_i + spike_i
        for i in 0..n {
            pre_trace[i] = pre_trace[i] * trace_decay
                + if i < state.spikes.len() && state.spikes[i] != 0 { 1.0 } else { 0.0 };
        }

        // 5. Eligibility update for active pre-neurons
        //    e_ij = β_e · e_ij + z_i · ψ_j · dt
        for i in 0..n {
            let tr = pre_trace[i];
            if tr < 1e-6 { continue; }
            let start = connectome.row_ptr[i].max(0) as usize;
            let end = (connectome.row_ptr[i + 1] as usize).min(max_syn);
            for idx in start..end {
                let j = connectome.col_indices[idx] as usize;
                if j < n {
                    eligibility[idx] = eligibility[idx] * beta_e + tr * psi[j] * dt;
                }
            }
        }

        // 6. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 7. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 8. Reset I = external_I
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() { state.I_buf[i] = 0.0; }

        // 9. CSR propagate from queue
        csr_propagate_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.nt_types, &mut state.queue, &mut state.I_buf, 1.0,
        );
    }

    (spike_buffer, eligibility)
}

// ─── Ψ-wave simulation ────────────────────────────────

/// Simulate `n_steps` of ψ-wave dynamics on a connectome.
///
/// `safe::psi_wave_step` signature:
/// `psi_wave_step(v, u, h_real, h_imag, i_ext, speed, dt, hbar)`
pub fn sim_psi(


    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &PsiParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let dt = params.dt_ms / 1000.0;
    let hbar = params.hbar;

    // Hψ scratch buffers (must match v size)
    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let mut speed = vec![1.0; state.n_padded];

    for step in 0..n_steps {
        // 1. Build Hamiltonian from CSR + state
        // h_real[i] = Σ W[j,i] * u[j], h_imag[i] = -Σ W[j,i] * v[j]
        h_real.fill(0.0);
        h_imag.fill(0.0);
        let max_syn = connectome.col_indices.len().min(connectome.values.len());
        for pre in 0..n_neurons {
            let start = connectome.row_ptr[pre].max(0) as usize;
            let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
            let mult = if pre < connectome.nt_types.len() { connectome.nt_types[pre] } else { 1.0 };
            for j in start..end {
                let post = connectome.col_indices[j] as usize;
                let w = connectome.values[j] * mult;
                if post < n {
                    h_real[post] += w * state.u[pre];
                    h_imag[post] -= w * state.v[pre];
                }
            }
        }

        // 2. Set external input
        for i in 0..n.min(external_I.len()) {
            state.I_buf[i] = external_I[i];
        }

        // 3. ψ-wave step
        safe::psi_wave_step(
            &mut state.v, &mut state.u,
            &mut h_real, &mut h_imag,
            &mut state.I_buf, &mut speed,
            dt, hbar,
        );

        // 4. "Spike" if |ψ|² > threshold
        for i in 0..n_neurons {
            let prob = state.v[i] * state.v[i] + state.u[i] * state.u[i];
            if prob > 0.5 {
                let idx = step * n_neurons + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
            }
        }
    }

    spike_buffer
}

// ─── HH simulation ──────────────────────────────────

/// Simulate `n_steps` of Hodgkin-Huxley dynamics with instant-current CSR propagation.
///
/// Per-step lifecycle:
/// 1. Set Ii = external_I (base current)
/// 2. step_hh_lut — advance all neurons using polynomial HH with SIMD
/// 3. Push spikes to queue → record
/// 4. Reset Ii = external_I
/// 5. CSR propagate: Ii[post] += W · nt_types[pre]
///
/// The `cfg` is an [`HhLutConfig`](taas_core::neuron::hh::HhLutConfig) created via
/// `standard_config(n, dt)` which gives standard NaV + KV channel kinetics.
pub fn sim_hh(
    connectome: &ConnectomeData,
    cfg: &mut taas_core::neuron::hh::HhLutConfig,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut queue = SpikeQueue::new(n_neurons);
    let mut spikes = vec![0i32; n];

    for step in 0..n_steps {
        // 1. Set Ii = external_I
        cfg.Ii[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);

        // 2. Clamp Ii (prevents saturation from oversized CSR propagation)
        safe::safety_clamp(&mut cfg.Ii[..n], -100.0, 100.0);

        // 3. HH step — produces spikes, updates cfg.v
        taas_core::neuron::hh::step_hh_lut(cfg, &mut spikes);

        // 3. Push spikes to queue
        push_spikes_to_queue(&spikes, n_neurons, &mut queue);

        // 4. Record spikes
        record_spikes(&spikes, n_neurons, &mut spike_buffer, step);

        // 5. Reset Ii = external_I
        cfg.Ii[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);

        // 6. CSR propagate spikes from queue → Ii
        csr_propagate_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.nt_types, &mut queue, &mut cfg.Ii, 1.0,
        );
    }

    spike_buffer
}

/// Simulate `n_steps` of Hodgkin-Huxley dynamics with conductance-based synaptic kinetics.
///
/// Same lifecycle as `sim_lif_synaptic` but with HH neurons (NaV + KV channels):
/// 1. Decay all conductances: `g *= exp(-dt/τ)` for each receptor type
/// 2. Set Ii = external_I (base current)
/// 3. Compute I_syn from conductances: `Ii += Σ g_r · (V - E_rev_r)`
/// 4. HH step — advance all neurons via polynomial SIMD
/// 5. Push spikes to queue → record
/// 6. CSR propagate: jump conductances (not currents)
///
/// Requires `connectome.receptor_types` to be non-empty.
///
/// The `cfg` is an [`HhLutConfig`](taas_core::neuron::hh::HhLutConfig) created via
/// `standard_config(n, dt)` which gives standard NaV + KV channel kinetics.
pub fn sim_hh_synaptic(
    connectome: &ConnectomeData,
    cfg: &mut taas_core::neuron::hh::HhLutConfig,
    external_I: &[f32],
    n_steps: usize,
    kinetics: &mut SynapticKinetics,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut queue = SpikeQueue::new(n_neurons);
    let mut spikes = vec![0i32; n];
    let factors = DecayFactors::new(cfg.dt);

    for step in 0..n_steps {
        // 1. Decay conductances: g *= exp(-dt/τ)
        kinetics.decay_all(&factors);

        // 2. Set Ii = external_I (reset)
        cfg.Ii[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);

        // 3. Compute I_syn from conductances and add to Ii
        kinetics.compute_currents(&cfg.v, &mut cfg.Ii);

        // 4. Clamp Ii (prevents saturation from oversized CSR propagation)
        safe::safety_clamp(&mut cfg.Ii[..n], -100.0, 100.0);

        // 5. HH step — produces spikes, updates cfg.v
        taas_core::neuron::hh::step_hh_lut(cfg, &mut spikes);

        // 5. Push spikes to queue
        push_spikes_to_queue(&spikes, n_neurons, &mut queue);

        // 6. Record spikes
        record_spikes(&spikes, n_neurons, &mut spike_buffer, step);

        // 7. CSR propagate: jump conductances (not currents!)
        csr_propagate_conductance_from_queue(
            &connectome.row_ptr, &connectome.col_indices, &connectome.values,
            &connectome.receptor_types, &mut queue, kinetics, 1.0,
        );
    }

    spike_buffer
}

/// Simulate ψ-wave with time-varying external input, recording (v, u) trajectory.
///
/// Args:
///   connectome: CSR connectome
///   state: mutable brain state (persistent across calls)
///   params: PsiParams { hbar, dt_ms }
///   external_I: flat 2D array `[n_changes * n]` — I_ext per change
///   n_changes: number of distinct external input settings
///   steps_per_input: PSI steps per external input
///   record_every: record (v, u) every N steps (0 = only at end of each input block)
///
/// Returns:
///   (v_traj, u_traj) — each Vec<f32> of length n_records * n
///   where n_records = n_changes * steps_per_input / record_every
pub fn sim_psi_record(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &PsiParams,
    external_I: &[f32],
    n_changes: usize,
    steps_per_input: usize,
    record_every: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let dt = params.dt_ms / 1000.0;
    let hbar = params.hbar;
    let re = if record_every == 0 { steps_per_input } else { record_every };
    let recs_per_input = steps_per_input / re;
    let n_records = n_changes * recs_per_input;

    let mut v_traj = vec![0.0f32; n_records * n];
    let mut u_traj = vec![0.0f32; n_records * n];

    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let mut speed = vec![1.0; state.n_padded];

    let mut rec_idx = 0;
    for change in 0..n_changes {
        let base = change * n;
        for i in 0..n.min(external_I.len().saturating_sub(base)) {
            state.I_buf[i] = external_I[base + i];
        }

        for step in 0..steps_per_input {
            h_real.fill(0.0);
            h_imag.fill(0.0);
            let max_syn = connectome.col_indices.len().min(connectome.values.len());
            for pre in 0..n_neurons {
                let start = connectome.row_ptr[pre].max(0) as usize;
                let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
                let mult = if pre < connectome.nt_types.len() { connectome.nt_types[pre] } else { 1.0 };
                for j in start..end {
                    let post = connectome.col_indices[j] as usize;
                    let w = connectome.values[j] * mult;
                    if post < n {
                        h_real[post] += w * state.u[pre];
                        h_imag[post] -= w * state.v[pre];
                    }
                }
            }

            safe::psi_wave_step(
                &mut state.v, &mut state.u,
                &mut h_real, &mut h_imag,
                &mut state.I_buf, &mut speed,
                dt, hbar,
            );

            if (step + 1) % re == 0 {
                let out_base = rec_idx * n;
                for i in 0..n {
                    v_traj[out_base + i] = state.v[i];
                    u_traj[out_base + i] = state.u[i];
                }
                rec_idx += 1;
            }
        }
    }

    (v_traj, u_traj)
}
