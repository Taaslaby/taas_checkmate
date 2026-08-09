//! # Measure/Phase parity — pillars-канон vs inline legacy
//!
//! Транш 2a: легаси raw-модули `kernels/src/{measure,phase,vnc}.rs` удалены;
//! пути `kernels::measure`, `kernels::phase`, `kernels::vnc` — compat-реэкспорты
//! pillars. Транш 3e: `safe.rs` удалён — функции зовутся напрямую из pillars.
//!
//! Здесь pillars-канон сравнивается с **настоящими inline-эталонами** из git HEAD
//! удалённых файлов (wc_parity-паттерн): бит-в-бит проверка, что pillars-канон
//! воспроизводит легаси-математику (включая SIMD/скаляр слияние и tail-циклы).

use pillars::kernels::measure::{
    band_energy, interference, interference_mean, phase_coherence, winding_matrix,
};

// ═══════════════════════════════════════════════════════════
//  Inline legacy references — verbatim из git HEAD
// ═══════════════════════════════════════════════════════════
mod legacy_measure {
    use core::f32::consts::PI;
    use wide::f32x8;

    unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
        let p = ptr.add(off);
        f32x8::from([
            *p,
            *p.add(1),
            *p.add(2),
            *p.add(3),
            *p.add(4),
            *p.add(5),
            *p.add(6),
            *p.add(7),
        ])
    }

    pub unsafe fn interference(
        output: *mut f32,
        phases_a: *const f32,
        phases_b: *const f32,
        na: usize,
        nb: usize,
    ) {
        let out = std::slice::from_raw_parts_mut(output, na * nb);
        let a = std::slice::from_raw_parts(phases_a, na);
        let b = std::slice::from_raw_parts(phases_b, nb);
        for i in 0..na {
            for j in 0..nb {
                out[i * nb + j] = (1.0 + (a[i] - b[j]).cos()) * 0.5;
            }
        }
    }

    pub unsafe fn interference_mean(
        output: *mut f32,
        phases_a: *const f32,
        phases_b: *const f32,
        n: usize,
    ) {
        let n8 = n & !7;
        let mut sum = f32x8::splat(0.0);
        for off in (0..n8).step_by(8) {
            let a = load8(phases_a, off);
            let b = load8(phases_b, off);
            let d = a - b;
            let da: [f32; 8] = d.to_array();
            let mut cv = [0.0f32; 8];
            for j in 0..8 {
                cv[j] = da[j].cos();
            }
            sum += f32x8::from(cv);
        }
        let arr: [f32; 8] = sum.to_array();
        let mut total: f32 = arr.iter().sum();
        if n8 < n {
            let a = std::slice::from_raw_parts(phases_a, n);
            let b = std::slice::from_raw_parts(phases_b, n);
            for i in n8..n {
                total += (a[i] - b[i]).cos();
            }
        }
        *output = (1.0 + total / n as f32) * 0.5;
    }

    pub unsafe fn phase_coherence(
        output: *mut f32,
        sin_buf: *const f32,
        cos_buf: *const f32,
        amplitudes: *const f32,
        n: usize,
    ) {
        let n8 = n & !7;
        let (mut ss, mut cc, mut sa) = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
        for off in (0..n8).step_by(8) {
            let si = load8(sin_buf, off);
            let co = load8(cos_buf, off);
            let a = load8(amplitudes, off);
            ss += a * si;
            cc += a * co;
            sa += a;
        }
        let ss_a: [f32; 8] = ss.to_array();
        let cc_a: [f32; 8] = cc.to_array();
        let sa_a: [f32; 8] = sa.to_array();
        let mut sum_s: f32 = ss_a.iter().sum();
        let mut sum_c: f32 = cc_a.iter().sum();
        let mut sum_a: f32 = sa_a.iter().sum();
        if n8 < n {
            let si = std::slice::from_raw_parts(sin_buf, n);
            let co = std::slice::from_raw_parts(cos_buf, n);
            let a = std::slice::from_raw_parts(amplitudes, n);
            for i in n8..n {
                sum_s += a[i] * si[i];
                sum_c += a[i] * co[i];
                sum_a += a[i];
            }
        }
        *output = (sum_s * sum_s + sum_c * sum_c).sqrt() / sum_a.max(1e-12);
    }

    pub unsafe fn band_energy(
        band_energies: *mut f32,
        amplitudes: *const f32,
        omegas: *const f32,
        band_edges: *const f32,
        n_bands: usize,
        n: usize,
    ) {
        let be = std::slice::from_raw_parts_mut(band_energies, n_bands);
        let a = std::slice::from_raw_parts(amplitudes, n);
        let o = std::slice::from_raw_parts(omegas, n);
        let edges = std::slice::from_raw_parts(band_edges, n_bands + 1);
        be.fill(0.0);
        for i in 0..n {
            let freq = o[i].abs();
            for b in 0..n_bands {
                if freq >= edges[b] && freq < edges[b + 1] {
                    be[b] += a[i];
                    break;
                }
            }
        }
    }

    pub unsafe fn winding_matrix(
        output: *mut f32,
        phases_a: *const f32,
        phases_b: *const f32,
        na: usize,
        nb: usize,
    ) {
        let out = std::slice::from_raw_parts_mut(output, na * nb);
        let a = std::slice::from_raw_parts(phases_a, na);
        let b = std::slice::from_raw_parts(phases_b, nb);
        for i in 0..na {
            for j in 0..nb {
                let d = a[i] - b[j];
                out[i * nb + j] = d.sin().atan2(d.cos()) / PI;
            }
        }
    }
}

mod legacy_phase {
    use core::f32::consts::TAU;
    use wide::f32x8;

    pub fn wrap_phase(p: f32) -> f32 {
        p - TAU * (p / TAU).round()
    }

    pub unsafe fn kuramoto_prepare(
        phases: *mut f32,
        cos_out: *mut f32,
        sin_out: *mut f32,
        n: usize,
    ) {
        let tau_s = f32x8::splat(TAU);
        let n8 = n & !7;

        for off in (0..n8).step_by(8) {
            let pv = f32x8::from([
                *phases.add(off),
                *phases.add(off + 1),
                *phases.add(off + 2),
                *phases.add(off + 3),
                *phases.add(off + 4),
                *phases.add(off + 5),
                *phases.add(off + 6),
                *phases.add(off + 7),
            ]);
            let wrapped = pv - tau_s * (pv / tau_s).round();
            // настоящий SIMD fast_sincos (как в легаси) — НЕ скалярный round/fma,
            // т.к. SIMD round = nearest-even, а f32::round = away-from-zero (1 ULP)
            let (si, co) = pillars::kernels::fast_sincos_simd(wrapped);
            let wr: [f32; 8] = wrapped.to_array();
            let co8: [f32; 8] = co.to_array();
            let si8: [f32; 8] = si.to_array();
            for i in 0..8 {
                *phases.add(off + i) = wr[i];
                *cos_out.add(off + i) = co8[i];
                *sin_out.add(off + i) = si8[i];
            }
        }

        for i in n8..n {
            let phi = wrap_phase(*phases.add(i));
            *phases.add(i) = phi;
            *cos_out.add(i) = phi.cos();
            *sin_out.add(i) = phi.sin();
        }
    }

    pub unsafe fn kuramoto_update(
        phases: *mut f32,
        omegas: *const f32,
        coupling: *const f32,
        dt: f32,
        n: usize,
    ) {
        let dt_s = f32x8::splat(dt);
        let n8 = n & !7;
        let tau_s = f32x8::splat(TAU);

        for off in (0..n8).step_by(8) {
            let pv = f32x8::from([
                *phases.add(off),
                *phases.add(off + 1),
                *phases.add(off + 2),
                *phases.add(off + 3),
                *phases.add(off + 4),
                *phases.add(off + 5),
                *phases.add(off + 6),
                *phases.add(off + 7),
            ]);
            let ov = f32x8::from([
                *omegas.add(off),
                *omegas.add(off + 1),
                *omegas.add(off + 2),
                *omegas.add(off + 3),
                *omegas.add(off + 4),
                *omegas.add(off + 5),
                *omegas.add(off + 6),
                *omegas.add(off + 7),
            ]);
            let cv = f32x8::from([
                *coupling.add(off),
                *coupling.add(off + 1),
                *coupling.add(off + 2),
                *coupling.add(off + 3),
                *coupling.add(off + 4),
                *coupling.add(off + 5),
                *coupling.add(off + 6),
                *coupling.add(off + 7),
            ]);
            let nv = pv + dt_s * (ov + cv);
            let nv = nv - tau_s * (nv / tau_s).round();
            let arr = nv.to_array();
            for (i, &value) in arr.iter().enumerate() {
                *phases.add(off + i) = value;
            }
        }
        for i in n8..n {
            let p = *phases.add(i);
            let nv = p + dt * (*omegas.add(i) + *coupling.add(i));
            *phases.add(i) = wrap_phase(nv);
        }
    }
}

fn assert_bits_eq(a: f32, b: f32, ctx: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{ctx}: pillars {a} (0x{:08x}) != legacy {b} (0x{:08x})",
        a.to_bits(),
        b.to_bits()
    );
}

fn phases_of(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 1.7 - 3.0).sin() * 3.0).collect()
}

// ─── measure ───

#[test]
fn measure_interference_parity() {
    let a = phases_of(6);
    let b = phases_of(5);
    let mut out_p = vec![0.0f32; 6 * 5];
    let mut out_l = vec![0.0f32; 6 * 5];
    interference(&mut out_p, &a, &b);
    unsafe {
        legacy_measure::interference(out_l.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 6, 5);
    }
    for i in 0..out_p.len() {
        assert_bits_eq(out_p[i], out_l[i], &format!("interference[{i}]"));
    }
}

#[test]
fn measure_interference_mean_parity() {
    for n in [8usize, 13, 7, 3] {
        let a = phases_of(n);
        let b = phases_of(n);
        let mut p = 0.0f32;
        let mut l = 0.0f32;
        interference_mean(&mut p, &a, &b);
        unsafe {
            legacy_measure::interference_mean(&mut l, a.as_ptr(), b.as_ptr(), n);
        }
        assert_bits_eq(p, l, &format!("interference_mean n={n}"));
    }
}

#[test]
fn measure_phase_coherence_parity() {
    for n in [8usize, 13, 7] {
        let sin_buf: Vec<f32> = phases_of(n).iter().map(|p| p.sin()).collect();
        let cos_buf: Vec<f32> = phases_of(n).iter().map(|p| p.cos()).collect();
        let amps: Vec<f32> = (0..n).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
        let mut p = 0.0f32;
        let mut l = 0.0f32;
        phase_coherence(&mut p, &sin_buf, &cos_buf, &amps);
        unsafe {
            legacy_measure::phase_coherence(
                &mut l,
                sin_buf.as_ptr(),
                cos_buf.as_ptr(),
                amps.as_ptr(),
                n,
            );
        }
        assert_bits_eq(p, l, &format!("phase_coherence n={n}"));
    }
}

#[test]
fn measure_band_energy_parity() {
    let n = 13;
    let amps: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.05).collect();
    let omegas: Vec<f32> = (0..n).map(|i| (i as f32 - 6.0).abs() * 2.0).collect();
    let edges = [0.0f32, 5.0, 10.0, 20.0, 1e9];
    let mut p = vec![0.0f32; 4];
    let mut l = vec![0.0f32; 4];
    band_energy(&mut p, &amps, &omegas, &edges);
    unsafe {
        legacy_measure::band_energy(
            l.as_mut_ptr(),
            amps.as_ptr(),
            omegas.as_ptr(),
            edges.as_ptr(),
            4,
            n,
        );
    }
    for i in 0..4 {
        assert_bits_eq(p[i], l[i], &format!("band_energy[{i}]"));
    }
}

#[test]
fn measure_winding_matrix_parity() {
    let a = phases_of(7);
    let b = phases_of(4);
    let mut out_p = vec![0.0f32; 7 * 4];
    let mut out_l = vec![0.0f32; 7 * 4];
    winding_matrix(&mut out_p, &a, &b);
    unsafe {
        legacy_measure::winding_matrix(out_l.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 7, 4);
    }
    for i in 0..out_p.len() {
        assert_bits_eq(out_p[i], out_l[i], &format!("winding[{i}]"));
    }
}

// ─── phase ───

#[test]
fn phase_kuramoto_prepare_parity() {
    for n in [8usize, 13, 7, 3] {
        let mut ph_p = phases_of(n);
        let mut ph_l = ph_p.clone();
        let mut co_p = vec![0.0f32; n];
        let mut si_p = vec![0.0f32; n];
        let mut co_l = vec![0.0f32; n];
        let mut si_l = vec![0.0f32; n];
        pillars::kernels::phase::kuramoto_prepare(&mut ph_p, &mut co_p, &mut si_p);
        unsafe {
            legacy_phase::kuramoto_prepare(
                ph_l.as_mut_ptr(),
                co_l.as_mut_ptr(),
                si_l.as_mut_ptr(),
                n,
            );
        }
        for i in 0..n {
            assert_bits_eq(ph_p[i], ph_l[i], &format!("prepare phase[{i}] n={n}"));
            assert_bits_eq(co_p[i], co_l[i], &format!("prepare cos[{i}] n={n}"));
            assert_bits_eq(si_p[i], si_l[i], &format!("prepare sin[{i}] n={n}"));
        }
    }
}

#[test]
fn phase_kuramoto_update_parity() {
    for n in [8usize, 13, 7] {
        let mut ph_p = phases_of(n);
        let mut ph_l = ph_p.clone();
        let omegas: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 * 0.2).collect();
        let coupling: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
        pillars::kernels::phase::kuramoto_update(&mut ph_p, &omegas, &coupling, 0.05);
        unsafe {
            legacy_phase::kuramoto_update(
                ph_l.as_mut_ptr(),
                omegas.as_ptr(),
                coupling.as_ptr(),
                0.05,
                n,
            );
        }
        for i in 0..n {
            assert_bits_eq(ph_p[i], ph_l[i], &format!("update phase[{i}] n={n}"));
        }
    }
}
