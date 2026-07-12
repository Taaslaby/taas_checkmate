//! Global Neuronal Workspace (GNW) — Baars 1988 / Dehaene 1998.
//!
//! Two linear layers with softmax competition:
//!   `gate:     n_neurons → workspace_size` — кто претендует на workspace
//!   `broadcast: workspace_size → n_neurons` — информация из workspace всем нейронам
//!   `output = r + broadcast(softmax(gate(r)))` — residual (ResNet-style)
//!
//! # Usage
//! ```
//! use karmac::gnw::GlobalWorkspace;
//! let mut gnw = GlobalWorkspace::new(512, 32);
//! let mut r = vec![0.0; 512];
//! gnw.forward(&mut r);
//! ```

use rand::Rng;
use rayon::prelude::*;

/// Global Neuronal Workspace.
pub struct GlobalWorkspace {
    /// gate matrix [workspace_size × n_neurons] (row-major)
    gate_w: Vec<f32>,
    /// broadcast matrix [n_neurons × workspace_size] (row-major)
    broadcast_w: Vec<f32>,
    n: usize,
    ws: usize,
    /// Pre-allocated delta buffer — avoids heap alloc in hot path.
    delta_buf: Vec<f32>,
}

impl GlobalWorkspace {
    /// Create GNW with `n_neurons` and `workspace_size`.
    /// Weights are Xavier-initialized (broadcast scaled to 0.01).
    pub fn new(n_neurons: usize, workspace_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier init for gate: scale = sqrt(6/(n + ws))
        let gate_scale = (6.0 / (n_neurons + workspace_size) as f32).sqrt();
        let gate_w: Vec<f32> = (0..n_neurons * workspace_size)
            .map(|_| rng.gen::<f32>() * 2.0 * gate_scale - gate_scale)
            .collect();

        // Xavier init * 0.01 for broadcast (small, don't dominate dynamics)
        let bc_scale = (6.0 / (n_neurons + workspace_size) as f32).sqrt() * 0.01;
        let broadcast_w: Vec<f32> = (0..workspace_size * n_neurons)
            .map(|_| rng.gen::<f32>() * 2.0 * bc_scale - bc_scale)
            .collect();

        Self { gate_w, broadcast_w, n: n_neurons, ws: workspace_size, delta_buf: vec![0.0; n_neurons] }
    }

    /// Create GNW with explicit weight matrices.
    pub fn from_weights(gate_w: Vec<f32>, broadcast_w: Vec<f32>) -> Self {
        let ws = gate_w.len() / broadcast_w.len();
        let n = broadcast_w.len() / ws;
        assert_eq!(gate_w.len(), n * ws);
        assert_eq!(broadcast_w.len(), ws * n);
        Self { gate_w, broadcast_w, n, ws, delta_buf: vec![0.0; n] }
    }

    /// Forward pass: `r += broadcast(softmax(gate(r)))`
    /// `r` is mutated in-place.
    pub fn forward(&mut self, r: &mut [f32]) {
        assert_eq!(r.len(), self.n);

        // gate: r (n) → gate_w (ws × n) → gates (ws)
        // gates = gate_w @ r
        let mut gates = vec![0.0f32; self.ws];
        for i in 0..self.ws {
            let row_start = i * self.n;
            let row = &self.gate_w[row_start..row_start + self.n];
            gates[i] = row.iter().zip(r.iter()).map(|(w, x)| w * x).sum();
        }

        // softmax on gates (numerically stable)
        let max_gate = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for g in gates.iter_mut() {
            *g = (*g - max_gate).exp();
            sum += *g;
        }
        let inv_sum = 1.0 / sum;
        for g in gates.iter_mut() {
            *g *= inv_sum;
        }

        // broadcast: gates (ws) → broadcast_w (n × ws) → delta (n)
        // delta = broadcast_w @ gates — pre-allocated buffer, no heap alloc
        self.delta_buf
            .par_chunks_exact_mut(1)
            .enumerate()
            .for_each(|(i, d)| {
                let row_start = i * self.ws;
                let row = &self.broadcast_w[row_start..row_start + self.ws];
                d[0] = row.iter().zip(gates.iter()).map(|(w, g)| w * g).sum();
            });

        // residual: r += delta
        for (r_i, d_i) in r.iter_mut().zip(self.delta_buf.iter()) {
            *r_i += d_i;
        }
    }

    pub fn n_neurons(&self) -> usize { self.n }
    pub fn workspace_size(&self) -> usize { self.ws }

    pub fn gate_weights(&self) -> &[f32] { &self.gate_w }
    pub fn broadcast_weights(&self) -> &[f32] { &self.broadcast_w }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gnw_forward_shape() {
        let mut gnw = GlobalWorkspace::new(512, 32);
        let mut r = vec![0.1; 512];
        gnw.forward(&mut r);
        assert_eq!(r.len(), 512);
        // Output should differ from input (GNW changes it)
        assert!(r.iter().any(|&x| (x - 0.1).abs() > 1e-6));
    }

    #[test]
    fn gnw_forward_small() {
        let mut gnw = GlobalWorkspace::new(4, 2);
        let mut r = vec![0.5, -0.3, 0.1, 0.0];
        gnw.forward(&mut r);
        assert_eq!(r.len(), 4);
        // Should not NaN
        for &v in &r { assert!(!v.is_nan()); }
    }
}
