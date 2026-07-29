//! ⚙️ Engine Pillar: оркестратор параллельных SIMD-вычислений.
//!
//! # Send/Safe
//! Все сырые указатели кастуются в `usize` перед захватом в Rayon-замыкания,
//! потому что `*mut f32` и `*mut TensorBridge` не реализуют `Send`/`Sync`.
//! Внутри замыкания `usize` обратно кастуется в указатель.

use rayon::prelude::*;
use super::lane::SimdLane;
use super::context::SimdContext;
use super::storage::StorageBuffer;
use super::tensor_bridge::TensorBridge;
use super::models::CaPlasticityModel;
use super::models::VSNModel;
use super::Model;

/// ⚙️ Engine Pillar: оркестратор полного шага симуляции.
pub struct EngineOrchestrator {
    syn_currents: Vec<f32>,
    surrogates: Vec<f32>,
    spikes: Vec<f32>,
}

impl EngineOrchestrator {
    pub fn new(neuron_cap: usize, synapse_cap: usize) -> Self {
        Self {
            syn_currents: vec![0.0f32; synapse_cap],
            surrogates: vec![0.0f32; neuron_cap],
            spikes: vec![0.0f32; neuron_cap],
        }
    }

    /// Полный шаг симуляции: синапсы → scatter → нейроны → gather.
    ///
    /// # Safety
    /// Все указатели (`*mut TensorBridge`, `*mut f32` столбцов состояния)
    /// кастуются в `usize` для `Send`-совместимости с Rayon.
    #[allow(non_snake_case)]
    pub unsafe fn step<S: SimdLane>(
        &mut self,
        neurons: &mut StorageBuffer<VSNModel>,
        synapses: &mut StorageBuffer<CaPlasticityModel>,
        bridge: &mut TensorBridge,
        dt: f32,
    ) {
        type SP = CaPlasticityModel;
        type NP = VSNModel;

        let SZ = S::SIZE;
        let ncap = neurons.capacity();
        let scap = synapses.capacity();

        // Resize temporary buffers if needed
        if self.syn_currents.len() < scap { self.syn_currents.resize(scap, 0.0); }
        if self.surrogates.len() < ncap { self.surrogates.resize(ncap, 0.0); }
        if self.spikes.len() < ncap { self.spikes.resize(ncap, 0.0); }

        // Cast to usize for Send-safe capture across Rayon threads.
        // Steps 1,3 only READ from bridge — use *const. Steps 2,4 WRITE — use *mut.
        let syn_base = synapses.as_ptr() as usize;
        let neu_base = neurons.as_ptr() as usize;
        let bdg_ro = bridge as *const TensorBridge as usize;
        let bdg_rw = bridge as *mut TensorBridge as usize;

        // ─── ШАГ 1: Пластичность синапсов (Rayon parallel, read-only bridge) ───
        (0..scap).into_par_iter().step_by(SZ).for_each(move |offset| {
            let bridge = &*(bdg_ro as *const TensorBridge);
            let syn_ptr = syn_base as *const f32;
            let syn_state_base = syn_ptr.add(SP::N_PARAMS * scap) as *mut f32;
            let mut ctx = SimdContext::<S>::new_raw(syn_ptr, scap, offset, dt);
            ctx.load_i(&bridge.i_pre_bus[offset..]);
            let post_val = S::load_param(bridge.i_post_bus.as_ptr(), 0, scap, offset);
            ctx.state[3] = post_val;
            SP::step(&mut ctx);
            for si in 0..SP::N_STATE {
                ctx.save_state_channel(si, syn_state_base.add(si * scap));
            }
        });

        // ─── ШАГ 2: Scatter токов из синапсов ───
        self.syn_currents[..scap].fill(0.0);
        {
            let syn_ptr = syn_base as *const f32;
            (0..scap).step_by(SZ).for_each(|offset| {
                let ctx = SimdContext::<S>::new_raw(syn_ptr, scap, offset, dt);
                let w = SP::weight(&ctx);
                S::write_to_slice(w, &mut self.syn_currents[offset..offset + SZ]);
            });
            let bridge = &mut *(bdg_rw as *mut TensorBridge);
            bridge.accumulate_synaptic_currents(&self.syn_currents[..scap]);
        }

        // ─── ШАГ 3: Нейроны VSN (Rayon parallel, read-only bridge) ───
        (0..ncap).into_par_iter().step_by(SZ).for_each(move |offset| {
            let bridge = &*(bdg_ro as *const TensorBridge);
            let neu_ptr = neu_base as *const f32;
            let neu_state_base = neu_ptr.add(NP::N_PARAMS * ncap) as *mut f32;
            let mut ctx = SimdContext::<S>::new_raw(neu_ptr, ncap, offset, dt);
            ctx.load_i(&bridge.i_inj_bus[offset..]);
            NP::step(&mut ctx);
            for si in 0..NP::N_STATE {
                ctx.save_state_channel(si, neu_state_base.add(si * ncap));
            }
        });

        // ─── ШАГ 4: Gather выходов нейронов → шины синапсов ───
        {
            let neu_ptr = neu_base as *const f32;
            self.surrogates[..ncap].fill(0.0);
            self.spikes[..ncap].fill(0.0);
            (0..ncap).step_by(SZ).for_each(|offset| {
                let ctx = SimdContext::<S>::new_raw(neu_ptr, ncap, offset, dt);
                let surr = NP::surrogate_output(&ctx);
                let spk = ctx.spike_mask();
                S::write_to_slice(surr, &mut self.surrogates[offset..offset + SZ]);
                S::write_to_slice(spk, &mut self.spikes[offset..offset + SZ]);
            });
            let bridge = &mut *(bdg_rw as *mut TensorBridge);
            bridge.route_neurons_to_synapses(&self.surrogates[..ncap], &self.spikes[..ncap]);
        }
    }
}
