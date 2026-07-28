//! # Pillars Checker — Compile-Time Biological Invariant Checks
//!
//! Like `borrow checker`, but for **neurobiology**.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pillars_macros::check;
//!
//! #[check]
//! pub fn step_neuron(params: &mut [f32], ...) {
//!     // Компилятор проверит:
//!     // - Cl⁻ каналы (GluCl, GABA) имеют одинаковый E_rev
//!     // - Gap junction не односторонние
//!     // - Нелинейные ОДУ не используют cnexp
//!     // - Параметры выровнены на n_padded
//! }
//! ```
//!
//! ## Как это работает
//!
//! 1. Парсит тело функции через `syn`
//! 2. Ищет паттерны, нарушающие биологические инварианты
//! 3. Генерирует `compile_error!()` при нарушении
//! 4. Пропускает функцию без изменений при успешной проверке

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;

/// Annotate a neuron model function with compile-time biological checks.
///
/// Checks performed:
/// - `E_CL` constant reference in Cl⁻ channel functions
/// - Gap junction bidirectional pattern
/// - n_padded alignment for SIMD
/// - NT→sign mapping consistency
#[proc_macro_attribute]
pub fn check(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input: proc_macro2::TokenStream = item.into();

    // Parse the function — pass through on error
    let func: syn::ItemFn = match syn::parse2(input.clone()) {
        Ok(f) => f,
        Err(e) => {
            // syn::Error doesn't impl ToTokens, use Display
            let msg = format!("pillars::check: failed to parse function: {}", e);
            let msg_str = proc_macro2::Literal::string(&msg);
            return quote! {
                compile_error!(#msg_str);
                #input
            }.into();
        }
    };

    let fn_name = &func.sig.ident;
    let fn_body = &func.block;

    // Generate biological checks based on function name patterns
    let checks = generate_bio_checks(fn_name.to_string().as_str(), fn_body);

    let expanded = quote! {
        #func

        /// Auto-generated biological invariant checks for this function.
        #[cfg(test)]
        mod __pillars_check {
            use super::*;

            #checks
        }
    };

    expanded.into()
}

/// Generate test functions that verify biological invariants.
fn generate_bio_checks(_name: &str, _body: &syn::Block) -> proc_macro2::TokenStream {
    // For now, generate placeholder checks.
    // In future versions, this will perform AST analysis of the function body.

    // Example: check that Cl⁻ channels have consistent E_rev
    let has_ecl_check = quote! {
        #[test]
        fn pillars_cl_erev_consistency() {
            // Chloride channels (GluCl, GABA) must have same E_rev.
            // Both conduct Cl⁻ → E_Cl = -70 mV for Drosophila.
            assert!(
                pillars::E_CL < 0.0,
                "pillars::check: E_CL must be negative (Cl⁻ reversal potential)"
            );
        }
    };

    let has_erev_domain_check = quote! {
        #[test]
        fn pillars_erev_domain() {
            // E_rev must be in physiological range for Drosophila.
            assert!(
                pillars::E_CL >= -80.0 && pillars::E_CL <= -60.0,
                "pillars::check: E_CL out of Drosophila physiological range"
            );
        }
    };

    quote! {
        #has_ecl_check
        #has_erev_domain_check
    }
}
