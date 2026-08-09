//! # Pillars Macros — #[derive(NeuronModel)]
//!
//! Generates typed State/Params structs + Model impl with step bridge.
//!
//! Generated bridges resolve the owning `pillars` dependency with
//! `proc-macro-crate`, so renamed downstream dependencies work too.

extern crate proc_macro;
use proc_macro::TokenStream;

use proc_macro2::TokenStream as TS2;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::{quote, ToTokens};
use std::collections::{HashMap, HashSet};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Expr, Ident, ItemMod, Pat, Stmt};

enum Role {
    State,
    Param,
}

fn role(attrs: &[Attribute]) -> Option<Role> {
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
fn no_spike(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("no_spike"))
}

/// Strip common suffixes like "Model" from type names.
fn base_name(name: &Ident) -> Ident {
    let s = name.to_string();
    let stripped = s.strip_suffix("Model").unwrap_or(&s);
    Ident::new(stripped, name.span())
}

/// Resolve the `pillars` crate as seen by the crate invoking this derive.
///
/// `FoundCrate::Itself` is important for the zoo models compiled inside the
/// pillars crate; downstream users may rename the dependency in Cargo.toml.
fn pillars_path() -> Result<TS2, TS2> {
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

#[proc_macro_derive(NeuronModel, attributes(state, param, no_spike))]
pub fn derive_neuron_model(input: TokenStream) -> TokenStream {
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
    // Index enums: use the FULL model name (LIFModelParamIdx) so they never
    // collide with the hand-written short enums (IzhParamIdx, HHParamIdx, …).
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

    // Params from_ctx
    let p_from_ctx: TS2 = pf
        .iter()
        .enumerate()
        .map(|(i, id)| quote! { #id: ctx.p(#i), })
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

            #[inline(always)]
            fn step<S: #pillars::core::lane::SimdLane>(ctx: &mut #pillars::core::context::SimdContext<S>) {
                unsafe {
                    let mut state = #sn::<S>::from_ctx(ctx);
                    let params = #pn::<S>::from_ctx(ctx);
                    let mut sc = #pillars::core::compute::StepContext::new(
                        ctx.i(), ctx.pre(), ctx.post(), ctx.dtf(),
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

// ═══════════════════════════════════════════════════════════════
//  #[dual_neuron] — ONE source file → CPU SIMD + CUDA C++
// ═══════════════════════════════════════════════════════════════
//
// Applies to a `mod` that contains a `#[derive(NeuronModel)]` struct +
// `impl ComputeNeuron`. The macro:
//   1. Leaves the Rust code untouched (CPU SIMD path — zero changes).
//   2. Transpiles the body of `ComputeNeuron::step` into a CUDA C++
//      `__device__` function string, plus a generic SoA kernel.
//   3. Emits `pub const <MOD>_CUDA_SOURCE: &str` next to the module.
//
// The transpiler supports the pillars model subset:
//   - let bindings, assignments, arithmetic, unary minus
//   - SimdLane methods: exp (exact expf), fast_exp (SFU exp2f — MUFU.EX2),
//     cube/square/powi, mul_add (FFMA),
//     cmp_ge/cmp_gt, and/not, abs, max/min, clamp01, cond().then().else_()
//   - Cond-typed locals: `let is_refr = …cmp_gt(…).cond();` then
//     `is_refr.then(a).else_(b)` (the Cond wraps the 1.0/0.0 mask — a float
//     on the CUDA side, so the chain compiles to a ternary on the variable)
//   - S::splat/one/zero/vtrap/sigmoid/blend, ctx.dt()/i_syn,
//     with ctx.dt_simd() retained as a compatibility alias,
//     ctx.emit_spike()
//   - `params.X.to_array()[0]` — scalar extraction of a param lane → P[idx]
//   - ops: ohmic_current, ohmic_gated, membrane_depolarize,
//     leak_integrate, edge_spike, fire_reset (tuple returns → out-params)
//
// Shadowing contract: Rust `let v = ...` that re-binds an in-scope name
// (state field or earlier local) transpiles to a C++ ASSIGNMENT, because
// C++ has no same-scope shadowing. Consequence: the local and its
// `state.<field>` share ONE C++ slot, so a model must not read a shadowed
// local AFTER reassigning the corresponding state field (Rust would see
// the old local value, C++ the new slot value). Read `state.<field>`
// before shadowing, as the zoo models do (`let v = ...else_(state.v)`).

#[proc_macro_attribute]
pub fn dual_neuron(attr: TokenStream, item: TokenStream) -> TokenStream {
    let _ = attr;
    let input = parse_macro_input!(item as ItemMod);
    let mod_ident = &input.ident;

    // ── 1. Collect model struct fields + step fn ──────────────────
    let mut s_fields: Vec<Ident> = Vec::new();
    let mut p_fields: Vec<Ident> = Vec::new();
    let mut step_block: Option<syn::Block> = None;
    let mut model_name: Option<Ident> = None;

    for item in input
        .content
        .as_ref()
        .map(|(_, items)| items)
        .into_iter()
        .flatten()
    {
        if let syn::Item::Struct(s) = item {
            let derives: Vec<String> = s
                .attrs
                .iter()
                .filter(|a| a.path().is_ident("derive"))
                .flat_map(|a| {
                    match &a.meta {
                    syn::Meta::List(l) => l.parse_args_with(
                        syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated
                    ).ok(),
                    _ => None,
                }
                })
                .flatten()
                .map(|p| {
                    p.segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default()
                })
                .collect();
            if derives.iter().any(|d| d == "NeuronModel") {
                model_name = Some(s.ident.clone());
                for f in s.fields.iter() {
                    let id = f.ident.clone().expect("named field");
                    match role(&f.attrs) {
                        Some(Role::State) => s_fields.push(id),
                        Some(Role::Param) => p_fields.push(id),
                        None => panic!("#[dual_neuron] field `{}` needs #[state] or #[param]", id),
                    }
                }
            }
        }
        if let syn::Item::Impl(imp) = item {
            if let syn::Type::Path(tp) = &imp.self_ty.as_ref() {
                let trait_path = imp.trait_.as_ref().map(|(_, p, _)| p).cloned();
                let is_compute = trait_path
                    .as_ref()
                    .and_then(|p| p.segments.last())
                    .map(|segment| segment.ident == "ComputeNeuron")
                    .unwrap_or(false);
                if is_compute {
                    for it in &imp.items {
                        if let syn::ImplItem::Fn(f) = it {
                            if f.sig.ident == "step" {
                                step_block = Some(f.block.clone());
                            }
                        }
                    }
                }
                let _ = tp;
            }
        }
    }

    let model =
        model_name.expect("#[dual_neuron] mod must contain a #[derive(NeuronModel)] struct");
    let step_block =
        step_block.expect("#[dual_neuron] mod must contain `impl ComputeNeuron { fn step .. }`");
    assert!(!s_fields.is_empty(), "model needs #[state] fields");
    assert!(!p_fields.is_empty(), "model needs #[param] fields");

    // ── 2. Build field → index maps ───────────────────────────────
    let p_idx: HashMap<String, usize> = p_fields
        .iter()
        .enumerate()
        .map(|(i, id)| (id.to_string(), i))
        .collect();
    let ns = s_fields.len();
    let np = p_fields.len();
    let fn_suffix = model.to_string().trim_end_matches("Model").to_lowercase();
    assert!(
        !fn_suffix.is_empty(),
        "#[dual_neuron] model name must not be exactly `Model` (kernel name would be `_kernel`)"
    );
    let step_fn_name = format!("{}_step", fn_suffix);
    let kernel_fn_name = format!("{}_kernel", fn_suffix);

    // ── 3. Collect consts + helper fns (so the transpiler can inline them) ──
    // The pillars CPU API lets models factor gate/current math into local
    // helper `fn`s taking `const` struct params (HH's `compute_gate_state` +
    // `GATE_M`). Collecting them here lets the CUDA bridge inline those calls
    // at translation time, so the zoo equations stay in one place (the Rust
    // step) — nothing is duplicated into CUDA C++.
    let (helpers, scalars, struct_consts) = {
        let mut scalar_raws: HashMap<String, &Expr> = HashMap::new();
        let mut struct_raws: HashMap<String, &Expr> = HashMap::new();
        let mut helpers: HashMap<String, HelperDef> = HashMap::new();
        for item in input
            .content
            .as_ref()
            .map(|(_, items)| items)
            .into_iter()
            .flatten()
        {
            if let syn::Item::Const(c) = item {
                let name = c.ident.to_string();
                if matches!(c.expr.as_ref(), Expr::Struct(_)) {
                    struct_raws.insert(name, c.expr.as_ref());
                } else {
                    scalar_raws.insert(name, c.expr.as_ref());
                }
            }
            if let syn::Item::Fn(f) = item {
                let params = f
                    .sig
                    .inputs
                    .iter()
                    .map(|a| match a {
                        syn::FnArg::Typed(pt) => match pt.pat.as_ref() {
                            Pat::Ident(i) => i.ident.to_string(),
                            _ => panic!("#[dual_neuron] helper param must be a plain ident"),
                        },
                        _ => panic!("#[dual_neuron] helper `{}` takes no receiver", f.sig.ident),
                    })
                    .collect();
                helpers.insert(
                    f.sig.ident.to_string(),
                    HelperDef {
                        params,
                        body: (*f.block).clone(),
                    },
                );
            }
        }
        // Resolve scalar consts (they may reference each other, e.g.
        // `GATE_CAT_INV_K = 1.0 / GATE_CAT_K`).
        let mut scalars: HashMap<String, String> = HashMap::new();
        let mut struct_consts: HashMap<String, HashMap<String, ConstField>> = HashMap::new();
        let mut in_progress: HashSet<String> = HashSet::new();
        for name in scalar_raws.keys() {
            if !scalars.contains_key(name) {
                resolve_scalar_const(
                    name,
                    scalar_raws[name],
                    &scalar_raws,
                    &mut scalars,
                    &mut in_progress,
                );
            }
        }
        for (name, init) in &struct_raws {
            struct_consts.insert(
                name.clone(),
                collect_struct_const(name, init, &scalar_raws, &mut scalars, &mut in_progress),
            );
        }
        (helpers, scalars, struct_consts)
    };

    // ── 4. Transpile body ─────────────────────────────────────────
    // Rust allows shadowing (`let v = ...` after `state.v` is in scope), but
    // C++ does not — redeclaring `float v` inside one function is an error.
    // Track declared names: state fields are pre-declared, so a `let` with a
    // colliding name becomes an assignment (`v = ...`), not a redeclaration.
    let declared: HashSet<String> = s_fields.iter().map(|id| id.to_string()).collect();
    let mut ctx = Ctx {
        p_idx: &p_idx,
        scalars: &scalars,
        struct_consts: &struct_consts,
        helpers: &helpers,
        declared,
        subst: Vec::new(),
        consts: Vec::new(),
    };
    let body = step_block
        .stmts
        .iter()
        .map(|st| transpile_stmt(st, &mut ctx))
        .collect::<Vec<String>>()
        .join("\n");
    let state_decls = s_fields
        .iter()
        .enumerate()
        .map(|(i, id)| format!("    float {} = S[{}];", id, i))
        .collect::<Vec<_>>()
        .join("\n");
    let state_writes = s_fields
        .iter()
        .enumerate()
        .map(|(i, id)| format!("    S[{}] = {};", i, id))
        .collect::<Vec<_>>()
        .join("\n");

    let header = format!("// Auto-generated by #[dual_neuron] — pillars model `{model}`\n");
    let math = concat!(
        "// ── math ──\n",
        "__device__ __forceinline__ float vtrap(float x, float y) {\n",
        "    if (fabsf(x / y) < 1e-6f) { return y * (1.0f - __fdividef(x, (2.0f * y))); }\n",
        "    // __fdividef → MUFU.RCP SFU (2 SASS ops) vs ~15-instr IEEE div.\n",
        "    // expf stays exact — CPU vtrap uses `exp` (SimdLane), not fast_exp.\n",
        "    return __fdividef(x, (expf(x / y) - 1.0f));\n",
        "}\n",
    );
    let ops = concat!(
        "// ── ops/ mirrors (1:1 with pillars) — the LEGO bricks ──\n",
        "// pillars ops use mul_add (FMA) — mirror that exactly so CPU and GPU\n",
        "// round identically (a plain mul would drift ~1 ulp per call).\n",
        "// NOTE: vtrap (math section) uses __fdividef (SFU RCP) — tolerance-\n",
        "// matched, NOT bit-matched (CPU does IEEE division there).\n",
        "__device__ __forceinline__ float ohmic_current(float g, float v, float e) { return fmaf(g, v, -g * e); }\n",
        "__device__ __forceinline__ float ohmic_gated(float g_max, float v, float e, float gating) { return fmaf(g_max * gating, v, -(g_max * gating) * e); }\n",
        "__device__ __forceinline__ float membrane_depolarize(float v, float i_inj, float i_ion, float inv_cm, float dt) { return v + (i_inj - i_ion) * inv_cm * dt; }\n",
        "__device__ __forceinline__ float leak_integrate(float v, float v_rest, float i_syn, float inv_tau, float dt) { return v + (v_rest - v + i_syn) * inv_tau * dt; }\n",
        "__device__ __forceinline__ void edge_spike(float is_above, float was_below, float& spike, float& was_below_next) {\n",
        "    spike = ((is_above != 0.0f) && (was_below > 0.0f)) ? 1.0f : 0.0f;\n",
        "    was_below_next = (is_above != 0.0f) ? 0.0f : 1.0f;\n",
        "}\n",
        "__device__ __forceinline__ void fire_reset(float v, float v_thresh, float v_reset, float& v_new, float& spike) {\n",
        "    spike = (v >= v_thresh) ? 1.0f : 0.0f;\n",
        "    v_new = (spike != 0.0f) ? v_reset : v;\n",
        "}\n",
    );
    let step_fn = [
        "// ── zoo/ step (transpiled from ComputeNeuron::step) ──",
        &format!("__device__ void {step_fn_name}(float* S, const float* P, float I, float DT, float* SPIKE) {{"),
        &state_decls,
        &body,
        &state_writes,
        "}",
    ].join("\n");
    let kernel = [
        "// ── core/ generic SoA kernel (column-major, stride = capacity, like pillars) ──",
        "// Scalar args (dt, n, stride) arrive as a pointer to a 3-float device",
        "// buffer — enables CUDA-Graph replay with per-step dt updates via memcpy",
        "// (the graph captures the buffer pointer, not the values).",
        &format!("__global__ void {kernel_fn_name}(float* states, const float* params, const float* i_in, int* spikes, const float* args) {{"),
        "    const float dt = args[0];",
        "    const int n = (int)args[1];",
        "    const int stride = (int)args[2];",
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (i >= n) return;",
        &format!("    float S[{ns}];"),
        "    #pragma unroll",
        "    for (int k = 0; k < NS; k++) S[k] = states[k * stride + i];",
        &format!("    float P[{np}];"),
        "    #pragma unroll",
        "    for (int k = 0; k < NP; k++) P[k] = params[k * stride + i];",
        "    float spike = 0.0f;",
        &format!("    {step_fn_name}(S, P, i_in[i], dt, &spike);"),
        "    #pragma unroll",
        "    for (int k = 0; k < NS; k++) states[k * stride + i] = S[k];",
        "    spikes[i] = (spike != 0.0f) ? 1 : 0;",
        "}",
    ].join("\n").replace("NS", &format!("{ns}")).replace("NP", &format!("{np}"));
    let cuda = format!("\n{header}\n{math}\n{ops}\n{step_fn}\n{kernel}\n");

    // ── 4. Emit: untouched mod + CUDA source const ────────────────
    let const_ident = Ident::new(
        &format!("{}_CUDA_SOURCE", mod_ident.to_string().to_uppercase()),
        mod_ident.span(),
    );
    let cleaned = ItemMod {
        attrs: vec![],
        content: input.content,
        ..input
    };
    let expanded = quote! {
        #cleaned
        /// CUDA C++ source auto-generated by `#[dual_neuron]` from the same
        /// `ComputeNeuron::step` body that runs on CPU SIMD.
        pub const #const_ident: &str = #cuda;
    };
    expanded.into()
}

// ── transpiler context ──────────────────────────────────────────────
// Added so the CUDA bridge understands the pillars CPU API as written:
// helper `fn`s, `const` struct params (GateParams) and const-foldable `if`.

/// A field of a struct const: float → C++ expression string, bool → value.
enum ConstField {
    F(String),
    B(bool),
}

/// A local helper `fn` collected from the `#[dual_neuron]` mod.
#[derive(Clone)]
struct HelperDef {
    params: Vec<String>,
    body: syn::Block,
}

/// Shared transpilation state threaded through `transpile_*`.
struct Ctx<'a> {
    p_idx: &'a HashMap<String, usize>,
    scalars: &'a HashMap<String, String>,
    struct_consts: &'a HashMap<String, HashMap<String, ConstField>>,
    helpers: &'a HashMap<String, HelperDef>,
    /// Names already bound in the emitted C++ scope (state fields pre-declared);
    /// a `let` re-binding one becomes an assignment (no shadowing in C++).
    declared: HashSet<String>,
    /// Inlining stacks: S-typed helper params → their transpiled arg expression;
    /// struct-const params → the const name (for field folding).
    subst: Vec<HashMap<String, String>>,
    consts: Vec<HashMap<String, String>>,
}

impl<'a> Ctx<'a> {
    /// Resolve a bare ident to an inlined helper-param substitution, if any.
    fn subst_of(&self, name: &str) -> Option<String> {
        for s in self.subst.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    /// Resolve a bare ident to a struct-const name bound to a helper param, if any.
    fn const_of(&self, name: &str) -> Option<String> {
        for c in self.consts.iter().rev() {
            if let Some(v) = c.get(name) {
                return Some(v.clone());
            }
        }
        None
    }
}

/// Skip statements whose `#[cfg(…)]` is inactive (e.g. `#[cfg(feature =
/// "hh-precise-gates")] let decay = …;`). Feature flags are read from the
/// Cargo env of the crate being compiled (the macro runs during its build).
fn stmt_cfg_active(st: &Stmt) -> bool {
    let attrs: &[Attribute] = match st {
        Stmt::Local(l) => &l.attrs,
        // Item / expr statements don't carry `#[cfg]` in the models we support
        // (cfg lives on `let` statements).
        Stmt::Item(_) | Stmt::Expr(_, _) => &[],
        Stmt::Macro(m) => &m.attrs,
    };
    attrs
        .iter()
        .all(|a| !a.path().is_ident("cfg") || cfg_meta_active(&a.meta))
}

fn cfg_meta_active(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::NameValue(nv) if nv.path.is_ident("feature") => {
            if let Expr::Lit(l) = &nv.value {
                if let syn::Lit::Str(s) = &l.lit {
                    let feat = s.value();
                    let env = format!("CARGO_FEATURE_{}", feat.to_uppercase().replace('-', "_"));
                    return std::env::var_os(env).is_some();
                }
            }
            // Unknown form — keep the statement (conservative).
            true
        }
        syn::Meta::List(l) => {
            let name = l.path.get_ident().map(|i| i.to_string());
            let inner: Vec<syn::Meta> = l
                .parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                )
                .unwrap_or_default()
                .into_iter()
                .collect();
            match name.as_deref() {
                Some("not") => inner.len() == 1 && !cfg_meta_active(&inner[0]),
                Some("any") => inner.iter().any(cfg_meta_active),
                Some("all") => inner.iter().all(cfg_meta_active),
                _ => true, // unknown predicate — keep
            }
        }
        _ => true,
    }
}

// ── scalar / struct const resolution (pillars CPU API: `const GATE_M:
//    GateParams = GateParams { … }`) ──────────────────────────────────

fn resolve_scalar(
    e: &Expr,
    raws: &HashMap<String, &Expr>,
    scalars: &mut HashMap<String, String>,
    in_progress: &mut HashSet<String>,
) -> String {
    match e {
        Expr::Paren(p) => format!("({})", resolve_scalar(&p.expr, raws, scalars, in_progress)),
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => format!("{}f", f.base10_digits()),
            syn::Lit::Int(i) => format!("{}f", i.base10_digits()),
            syn::Lit::Bool(b) => b.value.to_string(),
            _ => panic!("unsupported literal in #[dual_neuron] const initializer"),
        },
        Expr::Unary(u) => match u.op {
            syn::UnOp::Neg(_) => {
                format!("-({})", resolve_scalar(&u.expr, raws, scalars, in_progress))
            }
            _ => panic!("unsupported unary in #[dual_neuron] const initializer"),
        },
        Expr::Binary(b) => {
            let op = match b.op {
                syn::BinOp::Add(_) => "+",
                syn::BinOp::Sub(_) => "-",
                syn::BinOp::Mul(_) => "*",
                syn::BinOp::Div(_) => "/",
                _ => panic!("unsupported binary op in #[dual_neuron] const initializer"),
            };
            format!(
                "({} {} {})",
                resolve_scalar(&b.left, raws, scalars, in_progress),
                op,
                resolve_scalar(&b.right, raws, scalars, in_progress)
            )
        }
        Expr::Path(p) => {
            let ident = p.path.get_ident().expect("const path must be an ident");
            let name = ident.to_string();
            if let Some(v) = scalars.get(&name) {
                v.clone()
            } else if let Some(raw) = raws.get(&name) {
                resolve_scalar_const(&name, raw, raws, scalars, in_progress)
            } else {
                panic!("unknown scalar const `{name}` in #[dual_neuron] const initializer")
            }
        }
        _ => panic!("unsupported expression in #[dual_neuron] const initializer"),
    }
}

fn resolve_scalar_const(
    name: &str,
    raw: &Expr,
    raws: &HashMap<String, &Expr>,
    scalars: &mut HashMap<String, String>,
    in_progress: &mut HashSet<String>,
) -> String {
    if let Some(v) = scalars.get(name) {
        return v.clone();
    }
    if !in_progress.insert(name.to_string()) {
        panic!("cyclic scalar const `{name}` in #[dual_neuron]");
    }
    let resolved = resolve_scalar(raw, raws, scalars, in_progress);
    in_progress.remove(name);
    scalars.insert(name.to_string(), resolved.clone());
    resolved
}

/// Parse `const GATE_M: GateParams = GateParams { v_off: -40.0, …, alpha_exp:
/// false };` into a field table: float fields → C++ expression strings, bool
/// fields → bool values (for `if`-folding).
fn collect_struct_const(
    name: &str,
    init: &Expr,
    raws: &HashMap<String, &Expr>,
    scalars: &mut HashMap<String, String>,
    in_progress: &mut HashSet<String>,
) -> HashMap<String, ConstField> {
    let se = match init {
        Expr::Struct(se) => se,
        _ => panic!("const `{name}` must be a struct literal to be a struct const"),
    };
    let mut fields = HashMap::new();
    for f in &se.fields {
        let fname = f.member.to_token_stream().to_string();
        let val = resolve_scalar(&f.expr, raws, scalars, in_progress);
        let fval = if val == "true" {
            ConstField::B(true)
        } else if val == "false" {
            ConstField::B(false)
        } else {
            ConstField::F(val)
        };
        fields.insert(fname, fval);
    }
    fields
}

// ── transpile_stmt ────────────────────────────────────────────────

fn transpile_stmt(st: &Stmt, ctx: &mut Ctx) -> String {
    if !stmt_cfg_active(st) {
        return String::new();
    }
    match st {
        Stmt::Local(l) => {
            let init = l.init.as_ref().expect("let with init");
            match &l.pat {
                Pat::Ident(id) => {
                    let name = id.ident.to_string();
                    // Local helper `fn` call → inline its body (pillars CPU API:
                    // models factor gate/current math into helpers). The body's
                    // statements are emitted inline; its tail expression becomes
                    // this binding's initializer.
                    if let Some(call) = call_expr(init.expr.as_ref()) {
                        let fname = call_name(init.expr.as_ref()).unwrap_or_default();
                        if let Some(helper) = ctx.helpers.get(&fname).cloned() {
                            let args: Vec<&Expr> = call.args.iter().collect();
                            assert_eq!(
                                args.len(),
                                helper.params.len(),
                                "#[dual_neuron] helper `{fname}` arg count mismatch"
                            );
                            let mut sub: HashMap<String, String> = HashMap::new();
                            let mut con: HashMap<String, String> = HashMap::new();
                            for (pname, arg) in helper.params.iter().zip(args) {
                                // Struct-const arg (e.g. `GATE_M`) → const binding;
                                // `params.field` folds through `struct_consts`.
                                if let Expr::Path(p) = arg {
                                    if let Some(ident) = p.path.get_ident() {
                                        if ctx.struct_consts.contains_key(&ident.to_string()) {
                                            con.insert(pname.clone(), ident.to_string());
                                            continue;
                                        }
                                    }
                                }
                                sub.insert(pname.clone(), transpile_expr(arg, ctx));
                            }
                            ctx.subst.push(sub);
                            ctx.consts.push(con);
                            let mut out = String::new();
                            let stmts = &helper.body.stmts;
                            let last = stmts.len().saturating_sub(1);
                            for s in &stmts[..last] {
                                out.push_str(&transpile_stmt(s, ctx));
                                out.push('\n');
                            }
                            let tail = match stmts.last() {
                                Some(Stmt::Expr(e, _)) => transpile_expr(e, ctx),
                                _ => panic!(
                                    "#[dual_neuron] helper `{fname}` must end with a tail expression"
                                ),
                            };
                            if ctx.declared.insert(name.clone()) {
                                out.push_str(&format!("    float {name} = {tail};"));
                            } else {
                                out.push_str(&format!("    {name} = {tail};"));
                            }
                            ctx.subst.pop();
                            ctx.consts.pop();
                            return out;
                        }
                    }
                    let rhs = transpile_expr(init.expr.as_ref(), ctx);
                    // Shadowing: reuse the slot instead of redeclaring (C++
                    // has no shadowing in the same scope).
                    if ctx.declared.insert(name.clone()) {
                        format!("    float {name} = {rhs};")
                    } else {
                        format!("    {name} = {rhs};")
                    }
                }
                Pat::Tuple(t) => {
                    let names: Vec<String> = t
                        .elems
                        .iter()
                        .map(|p| match p {
                            Pat::Ident(i) => i.ident.to_string(),
                            _ => panic!("tuple pattern element must be an ident"),
                        })
                        .collect();
                    let decl = names.join(", ");
                    match call_name(init.expr.as_ref()) {
                        Some(f) if f == "edge_spike" || f == "fire_reset" => {
                            // In C++ the tuple result becomes out-params.
                            // Declare ONLY names not already in scope (same
                            // shadowing rule as Pat::Ident — a re-bound name
                            // must not be redeclared).
                            let fresh: Vec<String> = names
                                .iter()
                                .filter(|n| ctx.declared.insert((*n).clone()))
                                .cloned()
                                .collect();
                            if fresh.is_empty() {
                                format!(
                                    "    {f}({args}, {decl});",
                                    args = call_args(init.expr.as_ref(), ctx)
                                )
                            } else {
                                format!(
                                    "    float {decl};\n    {f}({args}, {decl});",
                                    f = f,
                                    args = call_args(init.expr.as_ref(), ctx)
                                )
                            }
                        }
                        _ => panic!(
                            "tuple-returning call `{}` — only edge_spike/fire_reset are supported",
                            init.expr.to_token_stream(),
                        ),
                    }
                }
                _ => panic!("unsupported let pattern in #[dual_neuron]"),
            }
        }
        Stmt::Expr(e, _) => transpile_expr_stmt(e, ctx),
        Stmt::Item(_) => panic!("item statements are not supported inside #[dual_neuron] step"),
        Stmt::Macro(_) => panic!("macro statements are not supported inside #[dual_neuron] step"),
    }
}

fn transpile_expr_stmt(e: &Expr, ctx: &mut Ctx) -> String {
    match e {
        Expr::Assign(a) => {
            let lhs = transpile_lvalue(a.left.as_ref(), ctx);
            format!("    {} = {};", lhs, transpile_expr(a.right.as_ref(), ctx))
        }
        Expr::MethodCall(m)
            if expr_is_ident(m.receiver.as_ref(), "ctx") && m.method == "emit_spike" =>
        {
            let arg = transpile_expr(&m.args[0], ctx);
            format!("    *SPIKE = {};", arg)
        }
        _ => panic!("statement must be an assignment or ctx.emit_spike(...)"),
    }
}

// ── shadowing-aware statement transpilation ────────────────────────
// (Rust `let v = ...` re-binding becomes C++ assignment to the pre-
// declared state/local slot; see `transpile_stmt` above.)

fn transpile_lvalue(e: &Expr, _ctx: &mut Ctx) -> String {
    if let Expr::Field(f) = e {
        if expr_is_ident(f.base.as_ref(), "state") {
            return f.member.to_token_stream().to_string();
        }
    }
    panic!("lvalue must be `state.<field>`; params are read-only")
}

// ── transpile_expr ────────────────────────────────────────────────

fn transpile_expr(e: &Expr, ctx: &mut Ctx) -> String {
    match e {
        Expr::Paren(p) => format!("({})", transpile_expr(p.expr.as_ref(), ctx)),
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => format!("{}f", f.base10_digits()),
            syn::Lit::Int(i) => format!("{}f", i.base10_digits()),
            syn::Lit::Bool(b) => {
                if b.value {
                    "1.0f".to_string()
                } else {
                    "0.0f".to_string()
                }
            }
            syn::Lit::Str(s) => s.value(),
            _ => panic!("unsupported literal in #[dual_neuron]"),
        },
        Expr::Path(p) => {
            let name = p
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_else(|| {
                    panic!(
                        "unsupported path in #[dual_neuron]: {}",
                        p.path.to_token_stream()
                    )
                });
            // Inlined helper S-param (subst), then scalar const, then plain local.
            if let Some(s) = ctx.subst_of(&name) {
                s
            } else if let Some(s) = ctx.scalars.get(&name) {
                s.clone()
            } else {
                name
            }
        }
        Expr::If(i) => {
            // Const-foldable `if` (e.g. `let alpha = if params.alpha_exp { … }
            // else { … };`) — the condition is a const-struct field read on an
            // inlined helper param. Only the active branch is emitted.
            match eval_const_bool(i.cond.as_ref(), ctx) {
                Some(true) => transpile_block_expr(&i.then_branch, ctx),
                Some(false) => match &i.else_branch {
                    Some((_, e2)) => transpile_expr(e2, ctx),
                    None => "0.0f".to_string(),
                },
                None => panic!(
                    "non-const `if` condition in #[dual_neuron] (only const-foldable conditions, e.g. `params.alpha_exp`, are supported): {}",
                    i.cond.to_token_stream()
                ),
            }
        }
        Expr::Block(b) => transpile_block_expr(&b.block, ctx),
        Expr::Match(_)
        | Expr::Loop(_)
        | Expr::While(_)
        | Expr::ForLoop(_)
        | Expr::Closure(_)
        | Expr::Return(_) => {
            panic!("control flow is not supported in #[dual_neuron] step (keep it expression-only)")
        }
        Expr::Unary(u) => match u.op {
            syn::UnOp::Neg(_) => format!("-({})", transpile_expr(u.expr.as_ref(), ctx)),
            syn::UnOp::Not(_) => panic!("`!` is not supported in #[dual_neuron]; use `.not()`"),
            _ => panic!("unsupported unary op in #[dual_neuron]"),
        },
        Expr::Binary(b) => {
            let op = match b.op {
                syn::BinOp::Add(_) => "+",
                syn::BinOp::Sub(_) => "-",
                syn::BinOp::Mul(_) => "*",
                syn::BinOp::Div(_) => "/",
                _ => panic!("unsupported binary op in #[dual_neuron]"),
            };
            format!(
                "({} {} {})",
                transpile_expr(b.left.as_ref(), ctx),
                op,
                transpile_expr(b.right.as_ref(), ctx)
            )
        }
        Expr::Field(f) => {
            let member = f.member.to_token_stream().to_string();
            // 1. Const-bound helper param (`params.scale` where the helper was
            //    called with `GATE_M`) — folds FIRST, because it shadows the
            //    model `params` name inside an inlined helper.
            if let Expr::Path(base) = f.base.as_ref() {
                if let Some(ident) = base.path.get_ident() {
                    if let Some(const_name) = ctx.const_of(&ident.to_string()) {
                        if let Some(fields) = ctx.struct_consts.get(&const_name) {
                            return match fields.get(&member) {
                                Some(ConstField::F(s)) => s.clone(),
                                Some(ConstField::B(_)) => panic!(
                                    "bool field `{member}` of const `{const_name}` used as a float"
                                ),
                                None => panic!(
                                    "unknown field `{member}` in const `{const_name}` (GateParams has v_off, scale, alpha_c, beta_c, beta_volt, beta_div, alpha_exp, beta_sigmoid)"
                                ),
                            };
                        }
                    }
                }
            }
            if expr_is_ident(f.base.as_ref(), "state") {
                member
            } else if expr_is_ident(f.base.as_ref(), "params") {
                let idx = ctx.p_idx.get(&member).expect("unknown param field");
                format!("P[{idx}]")
            } else if expr_is_ident(f.base.as_ref(), "ctx") {
                match member.as_str() {
                    "i_syn" => "I".to_string(),
                    _ => panic!("unsupported ctx field `{member}`"),
                }
            } else {
                panic!("unsupported field access in #[dual_neuron]")
            }
        }
        Expr::MethodCall(m) => transpile_method(m, ctx),
        Expr::Call(c) => {
            if let Expr::Path(p) = c.func.as_ref() {
                let segs: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                // S::splat(x) / S::one() / S::zero() / S::vtrap(a, b)
                if segs.len() == 2 && segs[0] == "S" {
                    return match segs[1].as_str() {
                        "splat" => transpile_expr(&c.args[0], ctx),
                        "one" => "1.0f".to_string(),
                        "zero" => "0.0f".to_string(),
                        "vtrap" => format!(
                            "vtrap({}, {})",
                            transpile_expr(&c.args[0], ctx),
                            transpile_expr(&c.args[1], ctx)
                        ),
                        // σ(v) = 1/(1+exp(-(v - v_half)/k)) — mirrors CPU
                        // math::sigmoid's DEFAULT path (exact SimdLane::exp,
                        // so expf is the faithful fp32 counterpart). Under the
                        // `fast-exp` feature the CPU sigmoid becomes the poly
                        // (~1e-5) while CUDA stays expf (~1e-7) — parity tests
                        // run without the feature.
                        "sigmoid" => format!(
                            "(1.0f / (1.0f + expf(-(({}) - ({})) / ({}))))",
                            transpile_expr(&c.args[0], ctx),
                            transpile_expr(&c.args[1], ctx),
                            transpile_expr(&c.args[2], ctx)
                        ),
                        // S::blend(cond, t, f) → ternary on the mask value.
                        "blend" => format!(
                            "(({} != 0.0f) ? {} : {})",
                            transpile_expr(&c.args[0], ctx),
                            transpile_expr(&c.args[1], ctx),
                            transpile_expr(&c.args[2], ctx)
                        ),
                        _ => panic!("unsupported S::{} in #[dual_neuron]", segs[1]),
                    };
                }
                let fname = segs.last().cloned().unwrap_or_default();
                match fname.as_str() {
                    "ohmic_current" | "ohmic_gated" | "membrane_depolarize" | "leak_integrate" => {
                        return format!("{} ({})", fname, call_args(e, ctx));
                    }
                    _ => panic!("unsupported function call `{fname}` in #[dual_neuron] (inline it or use an ops primitive)"),
                }
            }
            panic!("call with non-path function")
        }
        Expr::Index(i) => {
            // params.X.to_array()[0] — scalar extraction of one lane (WC uses
            // it to read a scalar param as f32). On GPU that is just P[idx].
            let is_zero_idx = matches!(
                i.index.as_ref(),
                Expr::Lit(el) if matches!(&el.lit, syn::Lit::Int(l) if l.base10_digits() == "0")
            );
            if is_zero_idx {
                if let Expr::MethodCall(m) = i.expr.as_ref() {
                    if m.method == "to_array" {
                        if let Expr::Field(f) = m.receiver.as_ref() {
                            let member = f.member.to_token_stream().to_string();
                            if expr_is_ident(f.base.as_ref(), "params") {
                                if let Some(idx) = ctx.p_idx.get(&member) {
                                    return format!("P[{idx}]");
                                }
                            }
                        }
                    }
                }
            }
            panic!(
                "indexing is not supported in #[dual_neuron] (only `params.X.to_array()[0]`): {}",
                e.to_token_stream()
            )
        }
        _ => panic!(
            "unsupported expression in #[dual_neuron]: {}",
            e.to_token_stream()
        ),
    }
}

/// Transpile an `if`/block used as a single-expression value (the branches in
/// `let alpha = if params.alpha_exp { … } else { … };`).
fn transpile_block_expr(b: &syn::Block, ctx: &mut Ctx) -> String {
    if b.stmts.len() == 1 {
        if let Stmt::Expr(e, _) = &b.stmts[0] {
            return transpile_expr(e, ctx);
        }
    }
    panic!("if-branch / block must be a single expression in #[dual_neuron]")
}

/// Fold an `if` condition to a compile-time bool. Supported: literals, `.not()`,
/// and const-struct field reads on an inlined helper param (`params.alpha_exp`).
fn eval_const_bool(e: &Expr, ctx: &Ctx) -> Option<bool> {
    match e {
        Expr::Paren(p) => eval_const_bool(p.expr.as_ref(), ctx),
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Bool(b) => Some(b.value),
            _ => None,
        },
        Expr::Unary(u) => match u.op {
            syn::UnOp::Not(_) => eval_const_bool(u.expr.as_ref(), ctx).map(|b| !b),
            _ => None,
        },
        Expr::Field(f) => {
            if let Expr::Path(base) = f.base.as_ref() {
                if let Some(ident) = base.path.get_ident() {
                    if let Some(const_name) = ctx.const_of(&ident.to_string()) {
                        if let Some(fields) = ctx.struct_consts.get(&const_name) {
                            let member = f.member.to_token_stream().to_string();
                            if let Some(ConstField::B(b)) = fields.get(&member) {
                                return Some(*b);
                            }
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn transpile_method(m: &syn::ExprMethodCall, ctx: &mut Ctx) -> String {
    // Both names map to the same runner-supplied scalar timestep. `dt()` is
    // canonical for new models; `dt_simd()` remains source-compatible.
    if expr_is_ident(m.receiver.as_ref(), "ctx") && (m.method == "dt" || m.method == "dt_simd") {
        return "DT".to_string();
    }
    // S::splat(x) / S::one() / S::zero() / S::vtrap(a, b)
    if expr_is_ident(m.receiver.as_ref(), "S") {
        return match m.method.to_string().as_str() {
            "splat" => transpile_expr(&m.args[0], ctx),
            "one" => "1.0f".to_string(),
            "zero" => "0.0f".to_string(),
            "vtrap" => format!(
                "vtrap({}, {})",
                transpile_expr(&m.args[0], ctx),
                transpile_expr(&m.args[1], ctx)
            ),
            _ => panic!("unsupported S::{} in #[dual_neuron]", m.method),
        };
    }
    // cond().then(a).else_(b) — handled at the outer `else_` call.
    // The condition may be an inline `.cond()` chain OR a `Cond`-typed local
    // (`let is_refr = state.refr.cmp_gt(zero).cond(); … is_refr.then(a).else_(b)`
    // — see SAW). Cond<S> wraps the 1.0/0.0 mask, so a stored Cond is just a
    // float on the CUDA side.
    if m.method == "else_" {
        if let Expr::MethodCall(then_call) = m.receiver.as_ref() {
            if then_call.method == "then" {
                let cond = match then_call.receiver.as_ref() {
                    Expr::MethodCall(cond_call) if cond_call.method == "cond" => {
                        cond_call.receiver.as_ref()
                    }
                    recv => recv,
                };
                let cond = transpile_expr(cond, ctx);
                let a = transpile_expr(&then_call.args[0], ctx);
                let b = transpile_expr(&m.args[0], ctx);
                return format!("(({} != 0.0f) ? {} : {})", cond, a, b);
            }
        }
        panic!("else_() without cond().then() chain");
    }
    let recv = transpile_expr(m.receiver.as_ref(), ctx);
    let name = m.method.to_string();
    match name.as_str() {
        // Standalone `.cond()` assigned to a local (`let is_refr =
        // …cmp_gt(…).cond();`) — Cond<S> wraps the 1.0/0.0 mask, so on the
        // CUDA side the local IS the mask float (identity). Inline chains
        // `x.cond().then(a).else_(b)` are handled by the `else_` branch above.
        "cond" => recv,
        // SFU fast path: exp2f(x) = single MUFU.EX2 instruction (~22-bit
        // precision, ~2-3 SASS ops) vs expf()'s ~20-instr IEEE path. The
        // x·log2e multiply folds the base conversion into the SFU operand.
        // Mirrors CPU `fast_exp` (minimax poly) semantics: fast, approximate.
        "fast_exp" => format!("exp2f(({}) * 1.4426950408889634f)", recv),
        // Exact IEEE path (~1 ulp) — kept for `exp()` so models that need
        // full precision keep it; matches the wide::f32x8::exp() CPU side.
        "exp" => format!("expf({})", recv),
        "cube" => format!("(({recv}) * ({recv}) * ({recv}))"),
        "square" => format!("(({recv}) * ({recv}))"),
        "powi" => {
            // x.powi::<4>() — N arrives as a const generic argument
            let n: usize = m
                .turbofish
                .as_ref()
                .and_then(|t| t.args.first())
                .and_then(|g| {
                    // `let ... else` instead of nested matches — each guard
                    // early-returns None, the final match keeps the parse.
                    let syn::GenericArgument::Const(expr) = g else {
                        return None;
                    };
                    let syn::Expr::Lit(el) = expr else {
                        return None;
                    };
                    match &el.lit {
                        syn::Lit::Int(i) => i.base10_digits().parse().ok(),
                        _ => None,
                    }
                })
                .expect("powi::<N> needs integer N");
            assert!(
                n <= 16,
                "powi::<N>: N must be <= 16 (mirrors the CPU `powi` const-assert)"
            );
            match n {
                // Mirror the CPU `powi_impl` special cases exactly, keeping the
                // same multiply grouping (bit-identical rounding): 0→1.0f,
                // 1→x, 2→x², 3→((x·x)·x), 4→s=x·x; s·s (two multiplies).
                0 => "1.0f".to_string(),
                1 => format!("({recv})"),
                2 => format!("(({recv})*({recv}))"),
                3 => format!("((({recv})*({recv}))*({recv}))"),
                4 => format!("((({recv})*({recv}))*(({recv})*({recv})))"),
                // General case (5..=16): repeated multiplication. Semantically
                // exact; grouping differs from the CPU binary-exponentiation
                // loop only at last-bit rounding — acceptable for CUDA parity.
                _ => {
                    let mut s = format!("({recv})");
                    for _ in 1..n {
                        s = format!("(({s}) * ({recv}))");
                    }
                    s
                }
            }
        }
        "mul_add" => format!(
            "fmaf({recv}, {}, {})",
            transpile_expr(&m.args[0], ctx),
            transpile_expr(&m.args[1], ctx)
        ),
        "cmp_ge" => format!(
            "(({recv} >= {}) ? 1.0f : 0.0f)",
            transpile_expr(&m.args[0], ctx)
        ),
        "cmp_gt" => format!(
            "(({recv} > {}) ? 1.0f : 0.0f)",
            transpile_expr(&m.args[0], ctx)
        ),
        "and" => format!(
            "(({recv} != 0.0f) && ({} != 0.0f)) ? 1.0f : 0.0f",
            transpile_expr(&m.args[0], ctx)
        ),
        "not" => format!("(({recv} == 0.0f) ? 1.0f : 0.0f)"),
        "abs" => format!("fabsf({recv})"),
        "max" => format!("fmaxf({recv}, {})", transpile_expr(&m.args[0], ctx)),
        "min" => format!("fminf({recv}, {})", transpile_expr(&m.args[0], ctx)),
        "clamp01" => format!("fminf(fmaxf({recv}, 0.0f), 1.0f)"),
        _ => panic!("unsupported SimdLane method `.{name}` in #[dual_neuron]"),
    }
}

fn expr_is_ident(e: &Expr, ident: &str) -> bool {
    matches!(e, Expr::Path(p) if p.path.is_ident(ident))
}

fn call_name(e: &Expr) -> Option<String> {
    match e {
        Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
        Expr::Call(c) => call_name(c.func.as_ref()),
        _ => None,
    }
}

fn call_expr(e: &Expr) -> Option<&syn::ExprCall> {
    if let Expr::Call(c) = e {
        Some(c)
    } else {
        None
    }
}

fn call_args(e: &Expr, ctx: &mut Ctx) -> String {
    if let Expr::Call(c) = e {
        return c
            .args
            .iter()
            .map(|a| transpile_expr(a, ctx))
            .collect::<Vec<_>>()
            .join(", ");
    }
    panic!("expected call")
}
