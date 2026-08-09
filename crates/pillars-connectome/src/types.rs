//! Connectome simulation types, state structures, and parameter representations.

/// Neuron model type.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NeuronModel {
    LIF,
    Izhikevich,
    Rate,
    Gco,
    /// Hodgkin-Huxley — state lives in `NeuralPopulation<HHModel>`;
    /// `BrainState` fields are inert for this model (state is read from the
    /// population).
    Hh,
}

// ═══════════════════════════════════════════════════════════
//  SpikeQueue — lock-free SPSC circular buffer
// ═══════════════════════════════════════════════════════════

/// Lock-free single-producer single-consumer spike queue (power-of-2 circular buffer).
#[derive(Debug, Clone)]
pub struct SpikeQueue {
    pub(crate) buffer: Vec<u32>,
    pub(crate) head: usize,
    pub(crate) tail: usize,
    pub(crate) mask: usize,
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

    /// Push a spike ID.
    #[inline(always)]
    pub fn push(&mut self, spike_id: u32) {
        debug_assert!(
            self.len() < self.buffer.len(),
            "SpikeQueue overflow: len={} cap={}",
            self.len(),
            self.buffer.len()
        );
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

    /// Drain all queued spikes.
    #[inline(always)]
    pub fn drain(&mut self) {
        self.tail = self.head;
    }

    /// Number of spikes currently in the queue.
    pub fn len(&self) -> usize {
        self.head - self.tail
    }

    /// Reset the queue.
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainState {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub trace: Vec<f32>,
    pub I_buf: Vec<f32>,
    pub spikes: Vec<i32>,
    pub refractory: Vec<f32>,
    pub n_padded: usize,
    #[serde(skip)]
    pub n_neurons: usize,
    #[serde(skip)]
    pub queue: SpikeQueue,
    pub slow_state: Vec<f32>,
}

impl BrainState {
    pub fn new(n_neurons: usize, model: NeuronModel) -> Self {
        let n_padded = n_neurons.div_ceil(8) * 8;
        let (v_init, u_init) = match model {
            NeuronModel::LIF => (-58.0, 0.0),
            NeuronModel::Izhikevich => (-65.0, -13.0),
            NeuronModel::Rate => (0.0, 0.0),
            NeuronModel::Gco => (0.0, 0.0),
            // HH state lives in the population; BrainState is inert (v=-65
            // kept as a sensible default for readers that peek at `v`).
            NeuronModel::Hh => (-65.0, 0.0),
        };

        let queue_cap = n_neurons.max(4).next_power_of_two();

        Self {
            v: vec![v_init; n_padded],
            u: vec![u_init; n_padded],
            trace: vec![0.0; n_padded],
            slow_state: vec![0.5; n_padded],
            I_buf: vec![0.0; n_padded],
            spikes: vec![0; n_padded],
            refractory: vec![0.0; n_padded],
            n_padded,
            n_neurons,
            queue: SpikeQueue::new(queue_cap),
        }
    }

    pub fn permute_state(&mut self, perm: &[usize]) {
        crate::rcm::apply_permutation_to_state(&mut self.v, perm);
        crate::rcm::apply_permutation_to_state(&mut self.u, perm);
        crate::rcm::apply_permutation_to_state(&mut self.trace, perm);
        crate::rcm::apply_permutation_to_state(&mut self.I_buf, perm);
        crate::rcm::apply_permutation_to_state(&mut self.spikes, perm);
        crate::rcm::apply_permutation_to_state(&mut self.refractory, perm);
        crate::rcm::apply_permutation_to_state(&mut self.slow_state, perm);
    }

    pub fn reset_with_noise(&mut self, n_neurons: usize) {
        for i in 0..n_neurons {
            let noise = ((i as f32 * 0.577) % 1.0 - 0.5) * 2.0;
            self.v[i] += noise * 0.03;
        }
    }
}

/// Packed LIF parameters for the SIMD kernel.
#[derive(Debug, Clone)]
pub struct LifParams {
    pub packed: Vec<f32>,
    pub tau_ref: f32,
    pub dt_ms: f32,
}

impl LifParams {
    pub fn new(
        tau_m: &[f32],
        v_rest: &[f32],
        v_thresh: &[f32],
        v_reset: &[f32],
        dt_ms: f32,
        tau_ref: f32,
    ) -> Self {
        let n = tau_m.len();
        let n_padded = n.div_ceil(8) * 8;
        let mut packed = vec![0.0f32; n_padded * 5];
        packed[0] = dt_ms;
        for i in 0..n {
            packed[n_padded + i] = tau_m[i];
            packed[2 * n_padded + i] = v_rest[i];
            packed[3 * n_padded + i] = v_thresh[i];
            packed[4 * n_padded + i] = v_reset[i];
        }
        for i in n..n_padded {
            packed[n_padded + i] = tau_m[n - 1];
            packed[2 * n_padded + i] = v_rest[n - 1];
            packed[3 * n_padded + i] = v_thresh[n - 1];
            packed[4 * n_padded + i] = v_reset[n - 1];
        }
        Self {
            packed,
            tau_ref,
            dt_ms,
        }
    }

    pub fn packed_params(&self) -> &[f32] {
        &self.packed
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

/// Parameters for Izhikevich Dual Memory model.
#[derive(Debug, Clone)]
pub struct IzhikevichDualParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub dt_ms: f32,
    pub tau_ref: f32,
    pub tau_m: f32,
    pub w_m: f32,
}

/// Parameters for Rate model.
#[derive(Debug, Clone)]
pub struct RateParams {
    pub tau: f32,
    pub dt_ms: f32,
}

/// Parameters for graph-coupled oscillator model.
#[derive(Debug, Clone)]
pub struct GcoParams {
    pub dt_ms: f32,
}
