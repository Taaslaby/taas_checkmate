//! # ⬡ CSR sparse operations — propagation, Hamiltonian, STDP
//!
//! Порт `core::sparse`: propagation, Hamiltonian, rate-step, STDP.
//! Hamiltonian parallelized with rayon (cfg-gated).
//!
//! Ported from `toyon/c/engine_neuron.c`.
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | `topology` | `bound_row`/`bound_col`, `ValidatedCsrTopology`, `ValidatedCsrGraph` |
//! | `propagate` | `propagate_spikes` (+ topology-validated variant) |
//! | `dynamics` | Hamiltonian, full oscillator step, rate-based step |
//! | `plasticity` | STDP LTD + the `_with_topology` variants |
//! | `tests` | unit tests (test-only) |

mod dynamics;
mod plasticity;
mod propagate;
#[cfg(test)]
mod tests;
mod topology;

pub use dynamics::*;
pub use plasticity::*;
pub use propagate::*;
pub use topology::*;
