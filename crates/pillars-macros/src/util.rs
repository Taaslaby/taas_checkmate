//! Shared helpers for the `NeuronModel` derive and the `dual_neuron` transpiler.

use proc_macro2::TokenStream as TS2;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{Attribute, Ident};

/// Field role in a `#[derive(NeuronModel)]` struct: a state channel or a
/// parameter column.
pub enum Role {
    State,
    Param,
}

/// Classify a field attribute: `#[state]` / `#[param]`.
pub fn role(attrs: &[Attribute]) -> Option<Role> {
    for a in attrs {
        if a.path().is_ident("state") {
            return Some(Role::State);
        }
        if a.path().is_ident("param") {
            return Some(Role::Param);
        }
    }
    None
}

/// Detect `#[no_spike]` struct attribute → `HAS_SPIKE = false`.
///
/// Synapses, glia and muscle models don't emit spikes, but the runner
/// still needs to know to skip the spike-mask writeback.
pub fn no_spike(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("no_spike"))
}

/// Strip common suffixes like "Model" from type names.
pub fn base_name(name: &Ident) -> Ident {
    let s = name.to_string();
    let stripped = s.strip_suffix("Model").unwrap_or(&s);
    Ident::new(stripped, name.span())
}

/// Resolve the `pillars` crate as seen by the crate invoking this derive.
///
/// `FoundCrate::Itself` is important for the zoo models compiled inside the
/// pillars crate; downstream users may rename the dependency in Cargo.toml.
pub fn pillars_path() -> Result<TS2, TS2> {
    match crate_name("pillars") {
        Ok(FoundCrate::Itself) => Ok(quote!(crate)),
        Ok(FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, proc_macro2::Span::call_site());
            Ok(quote!(::#ident))
        }
        Err(_) => Err(quote! {
            compile_error!("pillars-macros could not resolve the `pillars` dependency; add `pillars` to [dependencies]");
        }),
    }
}
