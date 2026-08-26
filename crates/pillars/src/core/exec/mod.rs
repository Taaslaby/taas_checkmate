//! # ⚙️ Execution — contexts, typed API, runners
//!
//! How a model actually runs on a population of neurons: the SIMD context
//! that carries state/params/buses, the typed `ComputeNeuron` API, and the
//! two runners (`NeuralPopulation` and the legacy raw-pointer `run_neuron*`).
//!
//! | File | Abstraction | Safety |
//! |------|-------------|:------:|
//! | [`context`] | `SimdContext<S>` — state/params/buses; `MAX_STATES = 32` | ⚠️ unsafe |
//! | [`compute`] | `ComputeNeuron` + `StepContext` — typed model API | ✅ safe |
//! | [`population`] | `NeuralPopulation` — safe SoA runner (main entry) | ⚠️ unsafe inside |
//! | [`runner`] | `run_neuron*` — legacy raw-pointer runners (≤6 states) | 🔒 unsafe |

pub mod compute;
pub mod context;
pub mod population;
pub mod runner;
