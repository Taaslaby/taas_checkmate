//! # ⚙️ Core — Zero-Cost SIMD Engine
//!
//! **Boundary: Rust generics → Hardware instructions.**
//!
//! ```text
//! User code:          v.cmp_ge(v_thresh)  ← generic over S: SimdLane
//!     ↓
//! SimdLane trait      SimdLane::cmp_ge(self, other) -> Self
//!     ↓                          ↓
//! f32x8 impl           f32 impl
//! _mm256_cmp_ps       f32::cmp(&v, &thresh)
//!     ↓                          ↓
//! AVX2: vcmpps        scalar: ucomiss + setae
//! ```
//!
//! Primitive methods in `SimdLane` (arithmetic, compares, loads) compile to
//! **exactly one hardware instruction**; composite helpers (`signum`, `round`,
//! `hypot`, `cbrt`) are short branchless sequences. No branching on data.
//! No lookup tables. No runtime dispatch.
//!
//! ## Module Boundaries
//!
//! | Module | Abstraction | Contains |
//! |--------|-------------|----------|
//! | `arch.rs` | Arch | `SimdType`, `LANES`, `ALIGN`, `pad_lanes` |
//! | `buf.rs` | Memory | `AlignedVec` — aligned SIMD buffers |
//! | `lane.rs` | Trait | `SimdLane` — the interface to hardware |
//! | `lane_f32x8.rs` | AVX2 | `SimdLane for f32x8` — maps to `_mm256_*` |
//! | `lane_f32.rs` | Scalar | `SimdLane for f32` — scalar fallback |
//! | `cond.rs` | Blend | `Cond<S>` — `.then().else_()` chain |
//! | `context.rs` | Execution | `SimdContext<S>` — state + params + buses (ядро) |
//! | `compute.rs` | Typed API | `ComputeNeuron` + `StepContext` |
//! | `population.rs` | Runner | `NeuralPopulation` (макросы load/save, unbounded states) |
//! | `runner.rs` | Legacy | `run_neuron*` — raw-pointer runners (≤6 состояний по сигнатуре) |
//! | `util.rs` | Helpers | `load8`, `store8` — raw SIMD load/store |
//!
//! ## State-channel limit
//!
//! [`context::MAX_STATES`] — единственное место, где ограничивается число
//! состояний модели (32). Макросы `load_state_*` / `save_state_*` и
//! `NeuralPopulation` итерируют до `M::N_STATE` на этапе компиляции — новых
//! состояний не нужно трогать unsafe-код. Legacy `run_neuron*` принимают
//! ровно 6 указателей состояний по сигнатуре; для более сложных моделей
//! используйте `NeuralPopulation`.
//!
//! ## Safety Boundary
//!
//! - **`arch.rs` / `buf.rs`** — safe (type + memory helpers)
//! - **`lane.rs`** — trait is safe (callable from safe code)
//! - **`lane_f32x8.rs`** — `unsafe` blocks for `_mm256_*` intrinsics
//! - **`population.rs`** — `unsafe` blocks for unaligned SIMD loads from heap
//! - **`runner.rs`** — `unsafe` blocks for raw-pointer `run_neuron*`
//! - **`context.rs`** — `unsafe` blocks for `load_param` dereference
//! - **All higher layers** (`ops/`, `zoo/`, `api/`) — safe Rust
//!
//! The `unsafe` is isolated to 4 files. If it compiles, it's correct.

pub mod arch;
pub mod buf;
pub mod compute;
pub mod cond;
pub mod context;
pub mod lane;
pub mod lane_f32;
pub mod lane_f32x8;
pub mod population;
pub mod runner;
pub mod util;

pub use arch::{
    is_aligned, is_slice_aligned, num_chunks, pad_lanes, tail_len, SimdType, ALIGN, LANES,
};
pub use buf::{split_simd, split_simd_ro, AlignedVec};

// Consistency-тесты «SIMD vs scalar» вынесены из lane.rs в отдельный файл.
#[cfg(test)]
mod lane_tests;

pub use compute::{ComputeNeuron, StepContext};
pub use cond::Cond;
pub use context::SimdContext;
pub use lane::SimdLane;
pub use population::NeuralPopulation;
#[cfg(feature = "rayon")]
pub use runner::run_neuron_par;
pub use runner::{run_neuron, run_neuron_ext, run_neuron_slices, NeuronState};
