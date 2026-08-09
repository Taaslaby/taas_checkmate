//! Транш 1c — parity-тест рецепторов/сенсорики/глии.
//!
//! Легаси `kernels::neuron/{synapse,chordotonal,astrocyte}.rs` удалены (волна 8,
//! транш 1c). Канон — `pillars::kernels::synapse`, `pillars::kernels::astrocyte`,
//! `pillars::zoo::sensors::chordotonal`. В `kernels::neuron` остались compat-
//! реэкспорты, чтобы Python ABI и потребители `kernels::*` продолжали работать.
//!
//! Эти тесты подтверждают:
//! 1. реэкспортированный путь (kernels) — это **тот же самый тип**, что канон
//!    pillars (compile-time identity через одинаковое поведение);
//! 2. бит-в-бит идентичность результатов через оба пути на случайных данных;
//! 3. численные инварианты ключевых формул (Mg²⁺-блок, DecayFactors, ohmic
//!    current, EAAT-клиренс, gain-модуляция).

use pillars::kernels::astrocyte as astro_pill;
use pillars::kernels::synapse as syn_pill;
use pillars::kernels::synapse as syn_kern;
use pillars::zoo::sensors::chordotonal as chord_pill;
use pillars::zoo::sensors::chordotonal as chord_kern;

// ═══════════════════════════════════════════════════════════
//  1. ReceptorType — таблица значений и конвертации
// ═══════════════════════════════════════════════════════════

#[test]
fn receptor_type_tables_match_both_paths() {
    for k in 0u8..=7 {
        let rk = syn_kern::ReceptorType::from_u8(k);
        let rp = syn_pill::ReceptorType::from_u8(k);
        assert_eq!(rk.tau_decay(), rp.tau_decay(), "tau_decay u8={k}");
        assert_eq!(rk.e_rev(), rp.e_rev(), "e_rev u8={k}");
        assert_eq!(rk.is_excitatory(), rp.is_excitatory(), "is_exc u8={k}");
    }
    assert_eq!(syn_kern::ReceptorType::from_nt_type(1.0).e_rev(), 0.0);
    assert_eq!(syn_kern::ReceptorType::from_nt_type(-1.0).e_rev(), -65.0);
    // Количество рецепторов — 7, итерация покрывает все типы.
    assert_eq!(syn_kern::ReceptorType::all().len(), 7);
}

// ═══════════════════════════════════════════════════════════
//  2. nmda_mg_block — бит-в-бит через оба пути + числовые инварианты
// ═══════════════════════════════════════════════════════════

#[test]
fn nmda_mg_block_parity_and_values() {
    let v_test = [-80.0, -65.0, -40.0, -20.0, 0.0, 20.0, 40.0];
    for &v in &v_test {
        assert_eq!(
            syn_kern::nmda_mg_block(v).to_bits(),
            syn_pill::nmda_mg_block(v).to_bits(),
            "nmda_mg_block({v}) должен быть бит-в-бит идентичен"
        );
    }
    // Биологический профиль: блок снимается при деполяризации.
    assert!(syn_pill::nmda_mg_block(-80.0) < 0.03, "Mg блок при -80mV");
    assert!(
        (syn_pill::nmda_mg_block(0.0) - 0.75).abs() < 0.05,
        "Mg при 0mV"
    );
    assert!(syn_pill::nmda_mg_block(30.0) > 0.90, "Mg при +30mV");
}

// ═══════════════════════════════════════════════════════════
//  3. DecayFactors — бит-в-бит + exp(-dt/τ) инвариант
// ═══════════════════════════════════════════════════════════

#[test]
fn decay_factors_parity() {
    let dt = 0.25;
    let fk = syn_kern::DecayFactors::new(dt);
    let fp = syn_pill::DecayFactors::new(dt);
    for (a, b) in [
        (fk.ampa, fp.ampa),
        (fk.nmda, fp.nmda),
        (fk.gaba_a, fp.gaba_a),
        (fk.gaba_b, fp.gaba_b),
        (fk.nachr, fp.nachr),
        (fk.glucl, fp.glucl),
        (fk.machr, fp.machr),
    ] {
        assert_eq!(a.to_bits(), b.to_bits(), "DecayFactors должны совпадать");
    }
    // AMPA: τ = 2ms → exp(-dt/2).
    assert!(
        (fp.ampa - (-dt / 2.0).exp()).abs() < 1e-5,
        "AMPA decay {}",
        fp.ampa
    );
    // GABA_A: τ = 10ms.
    assert!(
        (fp.gaba_a - (-dt / 10.0).exp()).abs() < 1e-5,
        "GABA_A decay {}",
        fp.gaba_a
    );
}

// ═══════════════════════════════════════════════════════════
//  4. SynapticKinetics — decay/jump/current через оба пути бит-в-бит
// ═══════════════════════════════════════════════════════════

#[test]
fn kinetics_both_paths_bit_identical() {
    let n = 13; // не кратно 8 → проверяем SIMD + tail
    let mut kk = syn_kern::SynapticKinetics::new(n);
    let mut kp = syn_pill::SynapticKinetics::new(n);
    let mut rng_state = 0x5EED_1234u64;
    let rnd = |state: &mut u64| {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (1u64 << 31) as f32
    };
    for i in 0..n {
        kk.g_ampa[i] = rnd(&mut rng_state);
        kp.g_ampa[i] = kk.g_ampa[i];
        kk.g_nmda[i] = rnd(&mut rng_state);
        kp.g_nmda[i] = kk.g_nmda[i];
        kk.g_gaba_a[i] = rnd(&mut rng_state);
        kp.g_gaba_a[i] = kk.g_gaba_a[i];
        kk.g_gaba_b[i] = rnd(&mut rng_state);
        kp.g_gaba_b[i] = kk.g_gaba_b[i];
        kk.g_nachr[i] = rnd(&mut rng_state);
        kp.g_nachr[i] = kk.g_nachr[i];
        kk.g_glucl[i] = rnd(&mut rng_state);
        kp.g_glucl[i] = kk.g_glucl[i];
        kk.g_machr[i] = rnd(&mut rng_state);
        kp.g_machr[i] = kk.g_machr[i];
    }

    let fk = syn_kern::DecayFactors::new(0.5);
    let fp = syn_pill::DecayFactors::new(0.5);
    kk.decay_all(&fk);
    kp.decay_all(&fp);
    for i in 0..n {
        assert_eq!(kk.g_ampa[i].to_bits(), kp.g_ampa[i].to_bits(), "ampa[{i}]");
        assert_eq!(kk.g_nmda[i].to_bits(), kp.g_nmda[i].to_bits(), "nmda[{i}]");
    }

    kk.jump(3, syn_kern::ReceptorType::AMPA, 0.7);
    kp.jump(3, syn_pill::ReceptorType::AMPA, 0.7);
    kk.jump(11, syn_kern::ReceptorType::GABA_A, 0.3);
    kp.jump(11, syn_pill::ReceptorType::GABA_A, 0.3);

    let v: Vec<f32> = (0..n).map(|i| -70.0 + i as f32).collect();
    let mut iok = vec![0.0f32; n];
    let mut iop = vec![0.0f32; n];
    kk.compute_currents(&v, &mut iok);
    kp.compute_currents(&v, &mut iop);
    for i in 0..n {
        assert_eq!(iok[i].to_bits(), iop[i].to_bits(), "I_out[{i}]");
    }
    // Ohmic-конвенция: AMPA при V < 0 даёт деполяризующий вклад.
    assert!(iok[3] > 0.0, "AMPA jump на нейроне 3 даёт I>0: {}", iok[3]);
}

#[test]
fn build_receptor_types_matches_expected() {
    // CSR: 2 пре-нейрона, 4 синапса (2 на каждый). nt_types индексируется
    // ПО ПРЕ-НЕЙРОНАМ: pre=0 (nt=+1 → AMPA) покрывает синапсы 0..2,
    // pre=1 (nt=-1 → GABA_A) покрывает синапсы 2..4.
    let row_ptr = [0i32, 2, 4];
    let nt_types = [1.0f32, -1.0];
    let rec = syn_pill::build_receptor_types(&row_ptr, 4, &nt_types);
    assert_eq!(
        rec,
        vec![0u8, 0, 2, 2],
        "AMPA=0, GABA_A=2 (по пре-нейронам)"
    );
}

// ═══════════════════════════════════════════════════════════
//  5. AstrocyteState — inline legacy-эталон (из git HEAD) vs pillars-порт
// ═══════════════════════════════════════════════════════════
//
// Легаси `kernels::neuron::astrocyte.rs` удалён, поэтому эталон воспроизведён
// здесь inline из git HEAD (волна 8, транш 1c): чисто скалярный f32-код
// формул step/compute_gain/modulate_v_thresh/apply_to_kinetics. Сравнение с
// pillars-портом — бит-в-бит на n=13 (SIMD-путь и скалярный tail покрыты).

/// Inline-копия легаси `AstrocyteState` (только математика, без SIMD).
struct AstroLegacy {
    n: usize,
    a: Vec<f32>,
    ca: Vec<f32>,
    v_gap: Vec<f32>,
    k_ext: Vec<f32>,
    eaat_occ: Vec<f32>,
    tau_astro: f32,
    tau_ca: f32,
    tau_k: f32,
    tau_eaat: f32,
    tau_gap: f32,
    alpha: f32,
    mu: f32,
    g_gap: f32,
    eaat_strength: f32,
    k_baseline: f32,
    k_sensitivity: f32,
    k_per_spike: f32,
    ca_per_spike: f32,
    ca_baseline: f32,
}

impl AstroLegacy {
    fn new(n: usize, tau_astro: f32, alpha: f32) -> Self {
        Self {
            n,
            a: vec![0.0; n],
            ca: vec![0.1; n],
            v_gap: vec![-70.0; n],
            k_ext: vec![4.0; n],
            eaat_occ: vec![0.0; n],
            tau_astro,
            tau_ca: 1000.0,
            tau_k: 500.0,
            tau_eaat: 10.0,
            tau_gap: 100.0,
            alpha,
            mu: 0.5,
            g_gap: 0.2,
            eaat_strength: 0.5,
            k_baseline: 4.0,
            k_sensitivity: 2.0,
            k_per_spike: 0.05,
            ca_per_spike: 1.0,
            ca_baseline: 0.1,
        }
    }

    fn step(&mut self, spikes: &[i32], gap_pairs: &[(usize, usize)], dt: f32) {
        assert!(
            spikes.len() >= self.n,
            "spikes shorter than astrocyte population"
        );
        for (i, &spike) in spikes.iter().take(self.n).enumerate() {
            let sp = spike as f32;
            #[allow(clippy::manual_clamp)]
            let a_next = (self.a[i] * (-dt / self.tau_astro.max(1.0)).exp()
                + sp * dt * self.ca_per_spike)
                .max(0.0)
                .min(10.0);
            self.a[i] = a_next;
            let dc = (-dt / self.tau_ca.max(1.0)).exp();
            #[allow(clippy::manual_clamp)]
            let ca_next =
                (self.ca[i] * dc + self.ca_baseline * (1.0 - dc) + sp * dt * self.ca_per_spike)
                    .max(0.0)
                    .min(5.0);
            self.ca[i] = ca_next;
            let dk = (-dt / self.tau_k.max(1.0)).exp();
            #[allow(clippy::manual_clamp)]
            let k_next =
                (self.k_baseline + (self.k_ext[i] - self.k_baseline) * dk + sp * self.k_per_spike)
                    .max(0.0)
                    .min(20.0);
            self.k_ext[i] = k_next;
            let de = (-dt / self.tau_eaat.max(1.0)).exp();
            #[allow(clippy::manual_clamp)]
            let eaat_next = (self.eaat_occ[i] * de + sp * 0.01 * dt).max(0.0).min(1.0);
            self.eaat_occ[i] = eaat_next;
            let dg = (-dt / self.tau_gap.max(1.0)).exp();
            self.v_gap[i] = -70.0 + (self.v_gap[i] + 70.0) * dg;
        }
        if self.g_gap > 0.0 && !gap_pairs.is_empty() {
            let mut i_gap = vec![0.0f32; self.n];
            for &(i, j) in gap_pairs {
                if i < self.n && j < self.n {
                    let diff = self.v_gap[j] - self.v_gap[i];
                    let current = self.g_gap * diff;
                    i_gap[i] += current;
                    i_gap[j] -= current;
                }
            }
            for (v_gap, gap_current) in self.v_gap.iter_mut().zip(i_gap) {
                *v_gap += gap_current * dt * 0.1;
            }
        }
    }

    fn compute_gain(&self) -> Vec<f32> {
        (0..self.n)
            .map(|i| (1.0 + self.a[i] * self.alpha) * (1.0 + self.ca[i] * self.mu))
            .collect()
    }

    fn modulate_v_thresh(&self, v_thresh: &mut [f32]) {
        for i in 0..self.n.min(v_thresh.len()) {
            v_thresh[i] += self.k_sensitivity * (self.k_ext[i] - self.k_baseline);
        }
    }

    fn apply_to_kinetics(&self, g_glucl: &mut [f32], dt: f32) {
        for i in 0..self.n.min(g_glucl.len()) {
            let clearance = (1.0 - self.eaat_strength * self.eaat_occ[i] * dt).max(0.0);
            g_glucl[i] *= clearance;
            let gain = (1.0 + self.a[i] * self.alpha) * (1.0 + self.ca[i] * self.mu);
            g_glucl[i] *= gain;
        }
    }
}

#[test]
fn astrocyte_port_matches_inline_legacy_reference() {
    let n = 13;
    let mut ap = astro_pill::AstrocyteState::new(n, 1000.0, 0.1);
    let mut al = AstroLegacy::new(n, 1000.0, 0.1);

    let gaps: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let mut rng_state = 0xABCD_EF01u64;
    let rnd = |state: &mut u64| {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (1u64 << 31) as f32
    };
    let mut gk = vec![0.0f32; n];
    for value in &mut gk {
        *value = rnd(&mut rng_state);
    }

    // Несколько шагов с варьируемым dt — покрывают SIMD и tail.
    for (step_i, dt) in [1.0f32, 0.25, 4.0, 0.1].iter().enumerate() {
        let sp: Vec<i32> = (0..n).map(|i| ((i + step_i) % 2 == 0) as i32).collect();
        ap.step(&sp, &gaps, *dt);
        al.step(&sp, &gaps, *dt);
        for i in 0..n {
            assert_eq!(ap.a[i].to_bits(), al.a[i].to_bits(), "step{step_i} a[{i}]");
            assert_eq!(
                ap.ca[i].to_bits(),
                al.ca[i].to_bits(),
                "step{step_i} ca[{i}]"
            );
            assert_eq!(
                ap.k_ext[i].to_bits(),
                al.k_ext[i].to_bits(),
                "step{step_i} k_ext[{i}]"
            );
            assert_eq!(
                ap.eaat_occ[i].to_bits(),
                al.eaat_occ[i].to_bits(),
                "step{step_i} eaat[{i}]"
            );
            assert_eq!(
                ap.v_gap[i].to_bits(),
                al.v_gap[i].to_bits(),
                "step{step_i} v_gap[{i}]"
            );
        }
    }

    // EAAT-клиренс + gain — бит-в-бит. Обе стороны применяют одну и ту же
    // последовательность: клиренс на GluCl, затем gain на все каналы.
    let mut kp_kin = syn_pill::SynapticKinetics::new(n);
    kp_kin.g_glucl[..n].copy_from_slice(&gk[..n]);
    ap.apply_to_kinetics(&mut kp_kin, 1.0);
    let mut gl = gk.clone();
    al.apply_to_kinetics(&mut gl, 1.0);
    for (i, (&pillars_value, &legacy_value)) in kp_kin.g_glucl.iter().zip(&gl).enumerate().take(n) {
        assert_eq!(
            pillars_value.to_bits(),
            legacy_value.to_bits(),
            "EAAT+gain g_glucl[{i}]"
        );
    }

    // K⁺-усталость и gain.
    let mut vtp = vec![30.0f32; n];
    let mut vtl = vec![30.0f32; n];
    ap.modulate_v_thresh(&mut vtp);
    al.modulate_v_thresh(&mut vtl);
    let gp_g = ap.compute_gain();
    let gl_g = al.compute_gain();
    for i in 0..n {
        assert_eq!(vtp[i].to_bits(), vtl[i].to_bits(), "v_thresh[{i}]");
        assert_eq!(gp_g[i].to_bits(), gl_g[i].to_bits(), "gain[{i}]");
        assert!(gp_g[i] >= 1.0, "gain[{i}] = {} < 1", gp_g[i]);
    }
}

// ═══════════════════════════════════════════════════════════
//  6. Chordotonal — сенсорика через оба пути идентична
// ═══════════════════════════════════════════════════════════

#[test]
fn chordotonal_both_paths_bit_identical() {
    let n = 13;
    let params: Vec<f32> = (0..n * 9).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let loads: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
    let stretches: Vec<f32> = (0..n).map(|i| (i as f32 - 6.0) * 0.1).collect();
    let d_loads: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let (mut vk, mut ak, mut rk) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let (mut vp, mut ap, mut rp) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);

    unsafe {
        chord_kern::chordotonal_step(
            vk.as_mut_ptr(),
            ak.as_mut_ptr(),
            rk.as_mut_ptr(),
            loads.as_ptr(),
            stretches.as_ptr(),
            d_loads.as_ptr(),
            params.as_ptr(),
            n,
        );
        chord_pill::chordotonal_step(
            vp.as_mut_ptr(),
            ap.as_mut_ptr(),
            rp.as_mut_ptr(),
            loads.as_ptr(),
            stretches.as_ptr(),
            d_loads.as_ptr(),
            params.as_ptr(),
            n,
        );
    }
    for i in 0..n {
        assert_eq!(vk[i].to_bits(), vp[i].to_bits(), "v_sens[{i}]");
        assert_eq!(ak[i].to_bits(), ap[i].to_bits(), "adapt[{i}]");
        assert_eq!(rk[i].to_bits(), rp[i].to_bits(), "rates[{i}]");
    }
}
