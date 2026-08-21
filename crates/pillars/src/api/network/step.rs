use super::*;
use crate::api::compact::compact_spikes_into;
use crate::api::delay::DelayBuffer;
use crate::api::graph::MAX_DELAY_SLOTS;
#[cfg(not(feature = "rayon"))]
use crate::api::push::push_spikes;
#[cfg(feature = "rayon")]
use crate::api::push::push_spikes_par;
use crate::api::push::{tm_spike, PlasticityState};
use crate::api::synapse::{exp_waveform_factor, SynapseKinetics};
use crate::core::buf::AlignedVec;

impl Network {
    /// Advance one simulation step for the entire network.
    ///
    /// # Panics
    /// Panics if `build()` has not been called.
    pub fn step(&mut self) {
        let Some(ref graph) = self.graph else {
            panic!("Network::step() called before build()");
        };
        if self.total_neurons == 0 {
            self.step_counter += 1; // advance counter even for empty networks
            return;
        }

        let slot = self.step_counter % MAX_DELAY_SLOTS;

        // ── 1. Read currents from current delay slot ──
        // Safe: `i_in` and `delay` are different fields → split borrow.
        self.delay.read_slot(slot, self.i_in.as_mut_slice());

        // ── 2. Clear the current slot for future writes ──
        self.delay.clear_slot(slot);

        // ── 2b. Copy membrane voltage for the kinetic driving force and gap
        // junctions (semi-implicit: V at the start of the step, like the HH
        // derivimplicit path). ──
        let need_v = !self.kinetics.is_empty()
            || !self.gap_junctions.is_empty()
            || !self.graded_synapses.is_empty()
            || !self.sigmoid_graded.is_empty();
        if need_v {
            let v = self.v_scratch.as_mut_slice();
            for pop_id in 0..self.pops.len() {
                let start = self.pop_offsets[pop_id];
                let end = self.pop_offsets[pop_id + 1];
                if end == start {
                    continue;
                }
                let state0 = self.pops[pop_id].state_slice(0);
                v[start..end].copy_from_slice(&state0[..end - start]);
            }
        }

        // Kinetic synapses: decay state, apply arriving jumps, and add the
        // resulting conductance current `g·(erev − V)` on top of the static
        // input for this step.
        if !self.kinetics.is_empty() {
            for (k, d) in self.kinetics.iter_mut().zip(self.kinetic_delay.iter_mut()) {
                k.step(
                    d,
                    slot,
                    self.kin_scratch.as_mut_slice(),
                    self.v_scratch.as_slice(),
                    self.i_in.as_mut_slice(),
                    self.total_neurons,
                );
            }
        }

        // Gap junctions (electrical synapses): symmetric ohmic coupling
        // `I_i += g·(V_j − V_i)`, `I_j += g·(V_i − V_j)`, from the start-of-step
        // V (explicit in V — stable for g·dt ≪ 1).
        if !self.gap_junctions.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for &(a, b, g) in &self.gap_junctions {
                let dv = v[b as usize] - v[a as usize];
                i_in[a as usize] += g * dv;
                i_in[b as usize] -= g * dv;
            }
        }

        // Graded (analog) synapses: one-way ohmic coupling
        // `I_post += g·(V_pre − V_post)`.
        if !self.graded_synapses.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for &(pre, post, g) in &self.graded_synapses {
                let dv = v[pre as usize] - v[post as usize];
                i_in[post as usize] += g * dv;
            }
        }

        // Sigmoid graded synapses: stateful activation `s` driven by V_pre,
        // current `weight·g·s·(erev − V_post)` into the post neuron.
        if !self.sigmoid_graded.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for rt in &mut self.sigmoid_graded {
                rt.step(v, i_in);
            }
        }

        // ── 3. Step each population ──
        for pop_id in 0..self.pops.len() {
            let start = self.pop_offsets[pop_id];
            let end = self.pop_offsets[pop_id + 1];
            let pop_n = end - start;
            if pop_n == 0 {
                continue;
            }
            let pop = &mut self.pops[pop_id];
            pop.step(
                &self.i_in.as_slice()[start..end],
                &mut self.spikes_out[start..end],
            );
        }

        // ── 4. Compact spikes → active indices (reusable scratch) ──
        self.active_scratch.clear();
        compact_spikes_into(&self.spikes_out, &mut self.active_scratch);

        // ── 5. Push active spikes through graph ──
        // Thread-local accumulation lives in `push_spikes_par` (push.rs):
        // Short-term plasticity state for this push: per-edge R/U/last. The
        // push scales each delivered weight by the edge's current R·U.
        let mut plast = if self.plast_profiles.is_empty() && self.plast_types.is_empty() {
            None
        } else {
            Some(PlasticityState {
                profiles: &self.plast_profiles,
                plast_types: &self.plast_types,
                r: &mut self.plast_r,
                u: &mut self.plast_u,
                last: &mut self.plast_last,
            })
        };
        // each thread collects (post, slot, weight) triples for its chunk;
        // the caller scatters them into the delay buffer after join.
        // Delays ARE respected — each triple carries the target slot.
        #[cfg(feature = "rayon")]
        push_spikes_par(
            &self.active_scratch,
            graph,
            &mut self.delay,
            &mut self.kinetic_delay,
            self.step_counter,
            plast.as_mut(),
        );

        #[cfg(not(feature = "rayon"))]
        push_spikes(
            &self.active_scratch,
            graph,
            &mut self.delay,
            &mut self.kinetic_delay,
            self.step_counter,
            plast.as_mut(),
        );

        // ── 5b. Roll Poisson sources and scatter their spikes ──
        // Same minimum-1-step delay rule as push_spikes (a spike fired at step
        // `s` is delivered at step `s + max(delay, 1)`).
        for src in &mut self.poisson_sources {
            if src.p <= 0.0 || !poisson_spike(src.seed, 0, self.step_counter as u64, src.p) {
                continue;
            }
            let step = self.step_counter;
            for i in 0..src.outgoing.len() {
                let (post, w, d, st, pt) = src.outgoing[i];
                let mut w = w;
                if pt > 0 {
                    let p = &self.plast_profiles[(pt - 1) as usize];
                    w = tm_spike(p, &mut src.r[i], &mut src.u[i], &mut src.last[i], w, step);
                }
                let target_slot = (step + (d as usize).max(1)) % MAX_DELAY_SLOTS;
                if st == 0 {
                    self.delay.add(target_slot, post as usize, w);
                } else {
                    self.kinetic_delay[(st as usize) - 1].add(target_slot, post as usize, w);
                }
            }
        }

        // ── 6. Advance counter ──
        self.step_counter += 1;
    }
}

/// Per-neuron kinetic-synapse state for one registered profile.
///
/// The ODE systems match the NeuroML/LEMS component definitions and are
/// integrated exactly (exponential Euler on linear ODEs, jumps applied at the
/// step boundary):
///
/// * expOne: `g` decays; the jump is added to `g`.
/// * expTwo: `g` (decay) and `x` (rise); the jump is added to both;
///   current = `norm·(g − x)`.
/// * alpha: `g` (conductance) and `a` (auxiliary); the jump is added to `a`
///   and `g` integrates `dg/dt = (e·a − g)/τ`.
/// * expThree: `a` (shared rise), `b` (decay1), `c` (decay2); the jump is
///   distributed per-state and the conductance is
///   `gbase1·(b−a) + gbase2·(c−a)`.
pub(crate) struct KineticRuntime {
    erev: f32,
    kind: KinKind,
    /// expOne/expTwo decay accumulator; alpha conductance.
    g: AlignedVec,
    /// expTwo rise accumulator.
    x: AlignedVec,
    /// alpha auxiliary state.
    a: AlignedVec,
    /// expThree decay-1 accumulator.
    b: AlignedVec,
    /// expThree decay-2 accumulator.
    c: AlignedVec,
}

/// Precomputed per-step integration constants for one waveform kind.
#[derive(Clone, Copy)]
pub(crate) enum KinKind {
    ExpOne {
        a_decay: f32,
    },
    ExpTwo {
        a_rise: f32,
        a_decay: f32,
        norm: f32,
    },
    Alpha {
        r: f32,
        e_r_dt_tau: f32,
    },
    ExpThree {
        a_rise: f32,
        a_decay1: f32,
        a_decay2: f32,
        wf1: f32,
        wf2: f32,
        k_a: f32,
        gbase1: f32,
        gbase2: f32,
    },
}

impl KineticRuntime {
    pub(crate) fn new(k: SynapseKinetics, dt: f32) -> Self {
        let erev = k.erev();
        let kind = match k {
            SynapseKinetics::ExpOne { tau_decay, .. } => KinKind::ExpOne {
                a_decay: (-dt / tau_decay).exp(),
            },
            SynapseKinetics::ExpTwo {
                tau_rise,
                tau_decay,
                ..
            } => KinKind::ExpTwo {
                a_rise: (-dt / tau_rise).exp(),
                a_decay: (-dt / tau_decay).exp(),
                norm: exp_waveform_factor(tau_rise, tau_decay),
            },
            SynapseKinetics::Alpha { tau, .. } => {
                let r = (-dt / tau).exp();
                KinKind::Alpha {
                    r,
                    e_r_dt_tau: std::f32::consts::E * (dt / tau) * r,
                }
            }
            SynapseKinetics::ExpThree {
                tau_rise,
                tau_decay1,
                tau_decay2,
                gbase1,
                gbase2,
                ..
            } => {
                let wf1 = exp_waveform_factor(tau_rise, tau_decay1);
                let wf2 = exp_waveform_factor(tau_rise, tau_decay2);
                KinKind::ExpThree {
                    a_rise: (-dt / tau_rise).exp(),
                    a_decay1: (-dt / tau_decay1).exp(),
                    a_decay2: (-dt / tau_decay2).exp(),
                    wf1,
                    wf2,
                    k_a: (gbase1 * wf1 + gbase2 * wf2) / (gbase1 + gbase2),
                    gbase1,
                    gbase2,
                }
            }
        };
        Self {
            erev,
            kind,
            g: AlignedVec::new(0, 0.0),
            x: AlignedVec::new(0, 0.0),
            a: AlignedVec::new(0, 0.0),
            b: AlignedVec::new(0, 0.0),
            c: AlignedVec::new(0, 0.0),
        }
    }

    pub(crate) fn allocate(&mut self, n: usize) {
        match self.kind {
            KinKind::ExpOne { .. } => self.g = AlignedVec::new(n, 0.0),
            KinKind::ExpTwo { .. } => {
                self.g = AlignedVec::new(n, 0.0);
                self.x = AlignedVec::new(n, 0.0);
            }
            KinKind::Alpha { .. } => {
                self.g = AlignedVec::new(n, 0.0);
                self.a = AlignedVec::new(n, 0.0);
            }
            KinKind::ExpThree { .. } => {
                self.a = AlignedVec::new(n, 0.0);
                self.b = AlignedVec::new(n, 0.0);
                self.c = AlignedVec::new(n, 0.0);
            }
        }
    }

    /// Advance one step: read arriving jumps from `delay` slot, decay the
    /// state, and add the conductance current `g·(erev − V)` into `i_in`.
    pub(crate) fn step(
        &mut self,
        delay: &mut DelayBuffer,
        slot: usize,
        scratch: &mut [f32],
        v: &[f32],
        i_in: &mut [f32],
        n: usize,
    ) {
        delay.read_slot(slot, &mut scratch[..n]);
        delay.clear_slot(slot);
        let erev = self.erev;
        match self.kind {
            KinKind::ExpOne { a_decay } => {
                let g = self.g.as_mut_slice();
                for i in 0..n {
                    let gn = g[i] * a_decay + scratch[i];
                    g[i] = gn;
                    i_in[i] += gn * (erev - v[i]);
                }
            }
            KinKind::ExpTwo {
                a_rise,
                a_decay,
                norm,
            } => {
                let g = self.g.as_mut_slice();
                let x = self.x.as_mut_slice();
                for i in 0..n {
                    let gn = g[i] * a_decay + scratch[i];
                    let xn = x[i] * a_rise + scratch[i];
                    g[i] = gn;
                    x[i] = xn;
                    i_in[i] += norm * (gn - xn) * (erev - v[i]);
                }
            }
            KinKind::Alpha { r, e_r_dt_tau } => {
                let g = self.g.as_mut_slice();
                let a = self.a.as_mut_slice();
                for i in 0..n {
                    let a_old = a[i];
                    let an = a_old * r + scratch[i];
                    let gn = g[i] * r + e_r_dt_tau * a_old;
                    g[i] = gn;
                    a[i] = an;
                    i_in[i] += gn * (erev - v[i]);
                }
            }
            KinKind::ExpThree {
                a_rise,
                a_decay1,
                a_decay2,
                wf1,
                wf2,
                k_a,
                gbase1,
                gbase2,
            } => {
                let a = self.a.as_mut_slice();
                let b = self.b.as_mut_slice();
                let c = self.c.as_mut_slice();
                for i in 0..n {
                    let j = scratch[i];
                    let an = a[i] * a_rise + j * k_a;
                    let bn = b[i] * a_decay1 + j * wf1;
                    let cn = c[i] * a_decay2 + j * wf2;
                    a[i] = an;
                    b[i] = bn;
                    c[i] = cn;
                    let cur = gbase1 * (bn - an) + gbase2 * (cn - an);
                    i_in[i] += cur * (erev - v[i]);
                }
            }
        }
    }
}
