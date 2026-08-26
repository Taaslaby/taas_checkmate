//! # 🧱 Memory & Hardware — layout, alignment, raw SIMD access
//!
//! The physical layer of the engine: SIMD width detection, aligned buffers,
//! and raw 8-wide load/store helpers. Everything here is about **where data
//! lives and how it is touched**, not about neuron math (that is `lane/`).
//!
//! | File | Abstraction | Safety |
//! |------|-------------|:------:|
//! | [`arch`] | `SimdType`, `LANES`, `ALIGN`, `pad_lanes` | ✅ safe |
//! | [`buf`] | `AlignedVec` — aligned SIMD buffers | ⚠️ unsafe inside |
//! | [`util`] | `load8`/`store8`/`prefetch_*` — raw SIMD load/store | 🔒 unsafe |

pub mod arch;
pub mod buf;
pub mod util;
