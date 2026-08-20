//! # Pillars Macros — #[derive(NeuronModel)] + #[dual_neuron]
//!
//! - [`NeuronModel`](derive_macro@NeuronModel) generates typed State/Params
//!   structs + a `Model` impl with a step bridge for a Rust neuron model.
//! - [`dual_neuron`](attr@dual_neuron) additionally transpiles the step body
//!   to CUDA C++ so one source file drives both the CPU SIMD engine and the
//!   GPU backend.
//!
//! Generated bridges resolve the owning `pillars` dependency with
//! `proc-macro-crate`, so renamed downstream dependencies work too.
//!
//! # Implementation layout
//!
//! Proc-macro entry points must live at the crate root, so this file only
//! holds thin `#[proc_macro_derive]` / `#[proc_macro_attribute]` forwards.
//! The actual implementation lives in the [`derive`] and [`dual`] modules;
//! shared helpers are in [`util`].

use proc_macro::TokenStream;

mod derive;
mod dual;
mod util;

#[proc_macro_derive(NeuronModel, attributes(state, param, no_spike, no_clamp, broadcast))]
pub fn derive_neuron_model(input: TokenStream) -> TokenStream {
    derive::derive_neuron_model_impl(input)
}

#[proc_macro_attribute]
pub fn dual_neuron(attr: TokenStream, item: TokenStream) -> TokenStream {
    dual::dual_neuron_impl(attr, item)
}
