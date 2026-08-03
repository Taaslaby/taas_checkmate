//! # Pillars Macros — #[derive(NeuronModel)]
//!
//! Generates typed State/Params structs + Model impl with step bridge.
//!
//! IMPORTANT: Use `crate::` paths (not `$crate` or `::pillars::`).
//! In a proc-macro context, `$crate` resolves to the macro's crate (pillars-macros),
//! but the generated code is compiled in `pillars`. So we use `crate::` which
//! always resolves to the compiling crate.

extern crate proc_macro;
use proc_macro::TokenStream;

use proc_macro2::TokenStream as TS2;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Attribute, Ident};

enum Role { State, Param }

fn role(attrs: &[Attribute]) -> Option<Role> {
    for a in attrs {
        if a.path().is_ident("state") { return Some(Role::State); }
        if a.path().is_ident("param") { return Some(Role::Param); }
    }
    None
}

/// Detect `#[no_spike]` struct attribute → `HAS_SPIKE = false`.
///
/// Synapses, glia and muscle models don't emit spikes, but the runner
/// still needs to know to skip the spike-mask writeback.
fn no_spike(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("no_spike"))
}

/// Strip common suffixes like "Model" from type names.
fn base_name(name: &Ident) -> Ident {
    let s = name.to_string();
    let stripped = s
        .strip_suffix("Model")
        .unwrap_or(&s);
    Ident::new(stripped, name.span())
}

#[proc_macro_derive(NeuronModel, attributes(state, param, no_spike))]
pub fn derive_neuron_model(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let base = base_name(name);

    let fields = match &input.data {
        Data::Struct(ds) => &ds.fields,
        _ => panic!("NeuronModel requires a struct with named fields"),
    };

    let mut sf: Vec<Ident> = Vec::new();
    let mut pf: Vec<Ident> = Vec::new();

    for f in fields.iter() {
        let id = f.ident.clone().expect("named field");
        match role(&f.attrs) {
            Some(Role::State) => sf.push(id),
            Some(Role::Param) => pf.push(id),
            None => panic!("field `{}` needs #[state] or #[param]", id),
        }
    }

    assert!(!sf.is_empty(), "at least one #[state] field");
    assert!(!pf.is_empty(), "at least one #[param] field");

    let ns = sf.len();
    let np = pf.len();
    let has_spike = !no_spike(&input.attrs);

    // Type names: LIF → LIFState, LIFParams (not LIFModelState)
    let sn = Ident::new(&format!("{}State", base), name.span());
    let pn = Ident::new(&format!("{}Params", base), name.span());

    // Field definitions
    let s_def: TS2 = sf.iter().map(|id| quote! { pub #id: S, }).collect();
    let p_def: TS2 = pf.iter().map(|id| quote! { pub #id: S, }).collect();

    // State from_ctx / to_ctx
    let from_ctx: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! { #id: ctx.s(#i), }).collect();
    let to_ctx: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! { ctx.set_s(#i, self.#id); }).collect();

    // State from_raw / to_raw (scalar)
    let from_raw: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! { #id: *ptr.add(#i * stride + off), }).collect();
    let to_raw: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! { *ptr.add(#i * stride + off) = self.#id; }).collect();

    // State from_raw_simd / to_raw_simd
    let from_raw_simd: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! {
            #id: (ptr.add(#i * stride + off) as *const crate::core::arch::SimdType).read_unaligned(),
        }).collect();
    let to_raw_simd: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! {
            let p = ptr.add(#i * stride + off) as *mut crate::core::arch::SimdType;
            p.write_unaligned(self.#id);
        }).collect();

    // Params from_ctx
    let p_from_ctx: TS2 = pf.iter().enumerate()
        .map(|(i, id)| quote! { #id: ctx.p(#i), }).collect();

    // Params pack_row / pack_soa
    let pack_row: TS2 = pf.iter().map(|id| quote! { self.#id, }).collect();
    let pack_soa: TS2 = pf.iter().enumerate()
        .map(|(i, id)| quote! { buf[#i * n + i] = self.#id; }).collect();

    // Params from_raw_simd
    let p_from_raw_simd: TS2 = pf.iter().enumerate()
        .map(|(i, id)| quote! {
            #id: (ptr.add(#i * stride + off) as *const crate::core::arch::SimdType).read_unaligned(),
        }).collect();

    // Params from_raw (scalar)
    let p_from_raw: TS2 = pf.iter().enumerate()
        .map(|(i, id)| quote! { #id: *ptr.add(#i * stride + off), }).collect();

    // ─── Generate code ─────────────────────────────────────
    let expanded = quote! {

        // ─── State struct ───
        pub struct #sn<S> { #s_def }

        impl<S: crate::core::lane::SimdLane> #sn<S> {
            #[inline(always)]
            pub(crate) unsafe fn from_ctx(ctx: &crate::core::context::SimdContext<S>) -> Self {
                Self { #from_ctx }
            }
            #[inline(always)]
            pub(crate) fn to_ctx(self, ctx: &mut crate::core::context::SimdContext<S>) {
                #to_ctx
            }
        }

        impl #sn<f32> {
            #[inline(always)]
            pub(crate) unsafe fn from_raw(ptr: *const f32, off: usize, stride: usize) -> Self {
                Self { #from_raw }
            }
            #[inline(always)]
            pub(crate) unsafe fn to_raw(self, ptr: *mut f32, off: usize, stride: usize) {
                #to_raw
            }
        }

        impl #sn<crate::core::arch::SimdType> {
            #[inline(always)]
            pub(crate) unsafe fn from_raw_simd(ptr: *const f32, off: usize, stride: usize) -> Self {
                Self { #from_raw_simd }
            }
            #[inline(always)]
            pub(crate) unsafe fn to_raw_simd(self, ptr: *mut f32, off: usize, stride: usize) {
                #to_raw_simd
            }
        }

        // ─── Params struct ───
        pub struct #pn<S> { #p_def }

        impl<S: crate::core::lane::SimdLane> #pn<S> {
            #[inline(always)]
            pub(crate) unsafe fn from_ctx(ctx: &crate::core::context::SimdContext<S>) -> Self {
                Self { #p_from_ctx }
            }
        }

        impl #pn<f32> {
            #[inline]
            pub fn pack_soa(&self, n: usize) -> Vec<f32> {
                let mut buf = vec![0.0; #np * n];
                for i in 0..n { #pack_soa }
                buf
            }
            #[inline]
            pub fn pack_row(&self) -> [f32; #np] {
                [#pack_row]
            }
            #[inline(always)]
            pub(crate) unsafe fn from_raw(ptr: *const f32, off: usize, stride: usize) -> Self {
                Self { #p_from_raw }
            }
        }

        impl #pn<crate::core::arch::SimdType> {
            #[inline(always)]
            pub(crate) unsafe fn from_raw_simd(ptr: *const f32, off: usize, stride: usize) -> Self {
                Self { #p_from_raw_simd }
            }
        }

        // ─── Model impl with bridge to ComputeNeuron ───
        impl crate::zoo::Model for #name {
            const N_PARAMS: usize = #np;
            const N_STATE: usize = #ns;
            const HAS_SPIKE: bool = #has_spike;

            #[inline(always)]
            fn step<S: crate::core::lane::SimdLane>(ctx: &mut crate::core::context::SimdContext<S>) {
                unsafe {
                    let mut state = #sn::<S>::from_ctx(ctx);
                    let params = #pn::<S>::from_ctx(ctx);
                    let mut sc = crate::core::compute::StepContext::new(
                        ctx.i(), ctx.pre(), ctx.post(), ctx.dtf(),
                    );
                    <Self as crate::core::compute::ComputeNeuron>::step::<S>(
                        &mut state, &params, &mut sc,
                    );
                    state.to_ctx(ctx);
                    ctx.spike(sc.spike_mask());
                }
            }
        }

        impl #name {
            pub const PARAM_COLS: usize = #np;
            pub const STATE_COLS: usize = #ns;
        }
    };

    expanded.into()
}
