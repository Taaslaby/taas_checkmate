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
//! ## Layout
//!
//! The engine is split into three subfolders + this facade:
//!
//! | Module | Abstraction | Contains |
//! |--------|-------------|----------|
//! | [`lane`] | SIMD interface | `SimdLane` trait + `f32x8`/`f32` impls, `Cond`, consistency tests |
//! | [`memory`] | Memory & hardware | `arch` (`SimdType`, `LANES`, `ALIGN`), `buf` (`AlignedVec`), `util` (`load8`/`store8`) |
//! | [`exec`] | Execution | `context` (`SimdContext`, `MAX_STATES`), `compute` (typed API), `population` (`NeuralPopulation`), `runner` (legacy `run_neuron*`) |
//!
//! The flat module paths from before the split are **re-exported unchanged**
//! below (`core::lane::SimdLane`, `core::runner::run_neuron`, …) so all
//! existing callers keep compiling.
//!
//! ## State-channel limit
//!
//! [`exec::context::MAX_STATES`] — единственное место, где ограничивается
//! число состояний модели (32). Макросы `load_state_*` / `save_state_*` и
//! `NeuralPopulation` итерируют до `M::N_STATE` на этапе компиляции — новых
//! состояний не нужно трогать unsafe-код. Legacy `run_neuron*` принимают
//! ровно 6 указателей состояний по сигнатуре; для более сложных моделей
//! используйте `NeuralPopulation`.
//!
//! ## Safety Boundary
//!
//! - **`memory/arch.rs` / `memory/buf.rs`** — safe (type + memory helpers)
//! - **`lane/mod.rs`** — trait is safe (callable from safe code)
//! - **`lane/f32x8.rs`** — `unsafe` blocks for `_mm256_*` intrinsics
//! - **`exec/population.rs`** — `unsafe` blocks for unaligned SIMD loads from heap
//! - **`exec/runner.rs`** — `unsafe` blocks for raw-pointer `run_neuron*`
//! - **`exec/context.rs`** — `unsafe` blocks for `load_param` dereference
//! - **All higher layers** (`ops/`, `zoo/`, `api/`) — safe Rust
//!
//! The `unsafe` is isolated to 4 files. If it compiles, it's correct.

pub mod lane;
pub mod memory;
pub mod exec;

// ── Flat re-export surface (kept for backward compatibility) ──────────
// The physical modules now live in subfolders; the old `core::X` paths are
// re-exported here so nothing outside `core` needs to change.
pub use memory::arch;
pub use memory::buf;
pub use memory::util;
pub use exec::compute;
pub use exec::context;
pub use exec::population;
pub use exec::runner;
pub use lane::cond;

pub use arch::{
    is_aligned, is_slice_aligned, num_chunks, pad_lanes, tail_len, SimdType, ALIGN, LANES,
};
pub use buf::{split_simd, split_simd_ro, AlignedVec};

// Consistency-тесты «SIMD vs scalar» живут в lane/tests.rs.
pub use compute::{ComputeNeuron, StepContext};
pub use cond::Cond;
pub use context::SimdContext;
pub use lane::SimdLane;
pub use population::NeuralPopulation;
#[cfg(feature = "rayon")]
pub use runner::run_neuron_par;
pub use runner::{run_neuron, run_neuron_ext, run_neuron_slices, NeuronState};
