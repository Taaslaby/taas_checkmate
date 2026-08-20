//! `#[derive(NeuronModel)]` — typed State/Params structs + Model impl.
//!
//! Shared helpers (`role`, `no_spike`, `base_name`, `pillars_path`) live in
//! the [`util`](crate::util) module and are reused by the `dual_neuron`
//! transpiler.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TS2;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Ident};

use super::util::{base_name, is_broadcast, no_clamp, no_spike, pillars_path, role, Role};

pub fn derive_neuron_model_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let pillars = match pillars_path() {
        Ok(path) => path,
        Err(diagnostic) => return diagnostic.into(),
    };
    let name = &input.ident;
    let base = base_name(name);

    let fields = match &input.data {
        Data::Struct(ds) => &ds.fields,
        _ => panic!("NeuronModel requires a struct with named fields"),
    };

    let mut sf: Vec<Ident> = Vec::new();
    let mut pf: Vec<Ident> = Vec::new();
    // Parallel to `pf`: true if the param is `#[broadcast]` (constant across
    // the population → read as a scalar splat instead of a column load).
    let mut pf_bc: Vec<bool> = Vec::new();

    for f in fields.iter() {
        let id = f.ident.clone().expect("named field");
        match role(&f.attrs) {
            Some(Role::State) => sf.push(id),
            Some(Role::Param) => {
                pf.push(id);
                pf_bc.push(is_broadcast(&f.attrs));
            }
            None => panic!("field `{}` needs #[state] or #[param]", id),
        }
    }

    assert!(!sf.is_empty(), "at least one #[state] field");
    assert!(!pf.is_empty(), "at least one #[param] field");

    let ns = sf.len();
    let np = pf.len();
    let has_spike = !no_spike(&input.attrs);
    let clamp_i_syn = !no_clamp(&input.attrs);

    // Type names: LIF → LIFState, LIFParams (not LIFModelState)
    let sn = Ident::new(&format!("{}State", base), name.span());
    let pn = Ident::new(&format!("{}Params", base), name.span());
    // Index enums: use the FULL model name (LIFModelParamIdx). The old
    // hand-written short enums (IzhParamIdx, HHParamIdx, …) were removed —
    // these generated enums are now the single source of truth.
    let param_enum = Ident::new(&format!("{}ParamIdx", name), name.span());
    let state_enum = Ident::new(&format!("{}StateIdx", name), name.span());

    // Field definitions
    let s_def: TS2 = sf.iter().map(|id| quote! { pub #id: S, }).collect();
    let p_def: TS2 = pf.iter().map(|id| quote! { pub #id: S, }).collect();

    // State from_ctx / to_ctx
    let from_ctx: TS2 = sf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { #id: ctx.s(#i), })
        .collect();
    let to_ctx: TS2 = sf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { ctx.set_s(#i, self.#id); })
        .collect();

    // State from_raw / to_raw (scalar)
    let from_raw: TS2 = sf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { #id: *ptr.add(#i * stride + off), })
        .collect();
    let to_raw: TS2 = sf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { *ptr.add(#i * stride + off) = self.#id; })
        .collect();

    // State from_raw_simd / to_raw_simd
    let from_raw_simd: TS2 = sf.iter().enumerate()
        .map(|(i, id)| quote! {
            #id: (ptr.add(#i * stride + off) as *const #pillars::core::arch::SimdType).read_unaligned(),
        }).collect();
    let to_raw_simd: TS2 = sf
        .iter()
        .enumerate()
        .map(|(i, id)| {
            quote! {
                let p = ptr.add(#i * stride + off) as *mut #pillars::core::arch::SimdType;
                p.write_unaligned(self.#id);
            }
        })
        .collect();

    // Params from_ctx — `#[broadcast]` params read via `ctx.pb` (scalar splat,
    // 4 bytes) instead of `ctx.p` (full S::SIZE-lane column load).
    let p_from_ctx: TS2 = pf
        .iter()
        .enumerate()
        .map(|(i, id)| {
            if pf_bc[i] {
                quote! { #id: ctx.pb(#i), }
            } else {
                quote! { #id: ctx.p(#i), }
            }
        })
        .collect();

    // Params pack_row / pack_soa
    let pack_row: TS2 = pf.iter().map(|id| quote! { self.#id, }).collect();
    let pack_soa: TS2 = pf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { buf[#i * n + i] = self.#id; })
        .collect();

    // Params from_raw_simd
    let p_from_raw_simd: TS2 = pf.iter().enumerate()
        .map(|(i, id)| quote! {
            #id: (ptr.add(#i * stride + off) as *const #pillars::core::arch::SimdType).read_unaligned(),
        }).collect();

    // Params from_raw (scalar)
    let p_from_raw: TS2 = pf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { #id: *ptr.add(#i * stride + off), })
        .collect();

    // Variant order = column order, so implicit discriminants (0, 1, 2, …)
    // equal the column indices — no explicit `= isize` needed.
    let p_variants: TS2 = pf.iter().map(|id| quote! { #id, }).collect();
    let s_variants: TS2 = sf.iter().map(|id| quote! { #id, }).collect();

    // ─── Generate code ─────────────────────────────────────
    let expanded = quote! {

        // ─── State struct ───
        pub struct #sn<S> { #s_def }

        impl<S: #pillars::core::lane::SimdLane> #sn<S> {
            #[inline(always)]
            pub(crate) unsafe fn from_ctx(ctx: &#pillars::core::context::SimdContext<S>) -> Self {
                Self { #from_ctx }
            }
            #[inline(always)]
            pub(crate) fn to_ctx(self, ctx: &mut #pillars::core::context::SimdContext<S>) {
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

        impl #sn<#pillars::core::arch::SimdType> {
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

        impl<S: #pillars::core::lane::SimdLane> #pn<S> {
            #[inline(always)]
            pub(crate) unsafe fn from_ctx(ctx: &#pillars::core::context::SimdContext<S>) -> Self {
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

        impl #pn<#pillars::core::arch::SimdType> {
            #[inline(always)]
            pub(crate) unsafe fn from_raw_simd(ptr: *const f32, off: usize, stride: usize) -> Self {
                Self { #p_from_raw_simd }
            }
        }

        // ─── Param/State index enums — kill magic column numbers ───
        // `pop.fill_param(LIFModelParamIdx::Dt.into(), 0.5)` instead of
        // `pop.fill_param(0, 0.5)`.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        pub enum #param_enum {
            #p_variants
        }
        impl From<#param_enum> for usize {
            #[inline(always)]
            fn from(v: #param_enum) -> Self {
                v as usize
            }
        }

        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        pub enum #state_enum {
            #s_variants
        }
        impl From<#state_enum> for usize {
            #[inline(always)]
            fn from(v: #state_enum) -> Self {
                v as usize
            }
        }

        // ─── Model impl with bridge to ComputeNeuron ───
        impl #pillars::zoo::Model for #name {
            const N_PARAMS: usize = #np;
            const N_STATE: usize = #ns;
            const HAS_SPIKE: bool = #has_spike;
            const CLAMP_I_SYN: bool = #clamp_i_syn;

            #[inline(always)]
            fn step<S: #pillars::core::lane::SimdLane>(ctx: &mut #pillars::core::context::SimdContext<S>) {
                unsafe {
                    let mut state = #sn::<S>::from_ctx(ctx);
                    let params = #pn::<S>::from_ctx(ctx);
                    let mut sc = #pillars::core::compute::StepContext::new(
                        ctx.i(), ctx.pre(), ctx.post(), ctx.dt_scalar(),
                    );
                    <Self as #pillars::core::compute::ComputeNeuron>::step::<S>(
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

