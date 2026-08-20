// ═══════════════════════════════════════════════════════════
//  Linear decoder
// ═══════════════════════════════════════════════════════════

/// Stateful linear decoder: `out = W·states + bias`.
///
/// Holds `weights` (n_in × n_out, row-major) and `bias` (n_out). The decode
/// kernel is single-sourced from [`crate::kernels::signal::linear_decoder`].
pub struct LinearDecoder {
    n_inputs: usize,
    n_outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearDecoder {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        Self {
            n_inputs: n_in,
            n_outputs: n_out,
            weights: vec![0.0f32; n_in * n_out],
            bias: vec![0.0f32; n_out],
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Decode: out = W * states + bias.
    pub fn decode(&mut self, states: &[f32], out_buffer: &mut [f32]) {
        assert!(states.len() >= self.n_inputs);
        assert!(out_buffer.len() >= self.n_outputs);

        crate::kernels::signal::linear_decoder(
            &mut out_buffer[..self.n_outputs],
            &states[..self.n_inputs],
            &self.weights,
            &self.bias,
        );
    }
}
