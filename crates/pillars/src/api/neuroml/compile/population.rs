use super::*;

use crate::api::network::Network;
use crate::api::neuroml::parse::{Cell, CellSpec};
use crate::api::neuroml::GpuPopMeta;
use crate::api::population::PopulationId;
use crate::api::{ConductancePopulation, MarkovChannel, MarkovPopulation};
use crate::kernels::hh_channels::{gate_kinetics_to_markov, ChannelMeta};
use crate::zoo::neurons::{
    AdExModel, FHNModel, HHModel, HHModelParamIdx, HHModelStateIdx, Izhikevich2007Model,
    Izhikevich2007ModelParamIdx, IzhikevichModel, IzhikevichModelParamIdx, LIFModel,
    ADEX_CUDA_SOURCE, FHN_CUDA_SOURCE, HH_CUDA_SOURCE, IZHIKEVICH_2007_CUDA_SOURCE,
    IZHIKEVICH_CUDA_SOURCE, LIF_CUDA_SOURCE,
};
use crate::zoo::Model;

/// Injected-current scale (mV per nA) for a cell's model. Pillars' LIF and
/// AdEx are voltage-normalised (÷gL), so their `current_scale` is 1000/gL;
/// every other model takes current in its native units (scale 1.0).
pub(crate) fn current_scale(cell: &Cell) -> f32 {
    match &cell.spec {
        CellSpec::AdEx { current_scale, .. } => *current_scale,
        CellSpec::Iaf { current_scale, .. } => *current_scale,
        _ => 1.0,
    }
}

/// Add a population of the right model type and configure it from `cell`.
///
/// Returns the `PopulationId` plus the [`GpuPopMeta`] (which `#[dual_neuron]`
/// CUDA source/kernel + column counts) so GPU backends can compile this
/// population without re-deriving the erased model type.
pub(crate) fn add_population(
    net: &mut Network,
    cell: &Cell,
    size: usize,
    dt_ms: f32,
    celsius: f32,
) -> (PopulationId, GpuPopMeta) {
    match &cell.spec {
        CellSpec::Izh {
            v0,
            thresh,
            a,
            b,
            c,
            d,
        } => {
            let pid = net.add_population::<IzhikevichModel>(size);
            net.configure::<IzhikevichModel>(pid, |p| {
                p.fill_param(IzhikevichModelParamIdx::dt as usize, dt_ms);
                p.fill_param(IzhikevichModelParamIdx::a as usize, *a);
                p.fill_param(IzhikevichModelParamIdx::b as usize, *b);
                p.fill_param(IzhikevichModelParamIdx::c as usize, *c);
                p.fill_param(IzhikevichModelParamIdx::d as usize, *d);
                p.fill_param(IzhikevichModelParamIdx::v_thresh as usize, *thresh);
                p.fill_state(0, *v0);
                p.fill_state(1, b * v0);
                p.fill_state(2, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: IZHIKEVICH_CUDA_SOURCE,
                    kernel_name: "izhikevich_kernel",
                    ns: IzhikevichModel::N_STATE,
                    np: IzhikevichModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Izh2007 {
            v0,
            cap,
            k,
            vr,
            vt,
            vpeak,
            a,
            b,
            c,
            d,
        } => {
            let pid = net.add_population::<Izhikevich2007Model>(size);
            net.configure::<Izhikevich2007Model>(pid, |p| {
                p.fill_param(Izhikevich2007ModelParamIdx::dt as usize, dt_ms);
                p.fill_param(Izhikevich2007ModelParamIdx::cap as usize, *cap);
                p.fill_param(Izhikevich2007ModelParamIdx::k as usize, *k);
                p.fill_param(Izhikevich2007ModelParamIdx::vr as usize, *vr);
                p.fill_param(Izhikevich2007ModelParamIdx::vt as usize, *vt);
                p.fill_param(Izhikevich2007ModelParamIdx::vpeak as usize, *vpeak);
                p.fill_param(Izhikevich2007ModelParamIdx::a as usize, *a);
                p.fill_param(Izhikevich2007ModelParamIdx::b as usize, *b);
                p.fill_param(Izhikevich2007ModelParamIdx::c as usize, *c);
                p.fill_param(Izhikevich2007ModelParamIdx::d as usize, *d);
                p.fill_state(0, *v0);
                // Steady-state recovery: u = b·(v0 − vr).
                p.fill_state(1, *b * (*v0 - *vr));
            });
            (
                pid,
                GpuPopMeta {
                    source: IZHIKEVICH_2007_CUDA_SOURCE,
                    kernel_name: "izhikevich2007_kernel",
                    ns: Izhikevich2007Model::N_STATE,
                    np: Izhikevich2007Model::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Iaf {
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_ref,
            current_scale: _,
        } => {
            let pid = net.add_population::<LIFModel>(size);
            net.configure::<LIFModel>(pid, |p| {
                // LIF layout (see LIFBuilder): dt, inv_tau_m, v_rest, v_thresh,
                // v_reset, tau_ref; state v, refr.
                p.fill_param(0, dt_ms);
                p.fill_param(1, 1.0 / tau_m);
                p.fill_param(2, *v_rest);
                p.fill_param(3, *v_thresh);
                p.fill_param(4, *v_reset);
                p.fill_param(5, *tau_ref);
                p.fill_state(0, *v_rest);
                p.fill_state(1, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: LIF_CUDA_SOURCE,
                    kernel_name: "lif_kernel",
                    ns: LIFModel::N_STATE,
                    np: LIFModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Fhn {
            a,
            b,
            inv_tau_w,
            v_thresh,
            bias: _,
            v0,
            w0,
        } => {
            let pid = net.add_population::<FHNModel>(size);
            net.configure::<FHNModel>(pid, |p| {
                // FHN layout: dt, inv_tau_w, a, b, v_thresh; state v, w, was_below.
                p.fill_param(0, dt_ms);
                p.fill_param(1, *inv_tau_w);
                p.fill_param(2, *a);
                p.fill_param(3, *b);
                p.fill_param(4, *v_thresh);
                p.fill_state(0, *v0);
                p.fill_state(1, *w0);
                p.fill_state(2, 1.0); // was_below = true (edge detection ready)
            });
            (
                pid,
                GpuPopMeta {
                    source: FHN_CUDA_SOURCE,
                    kernel_name: "fhn_kernel",
                    ns: FHNModel::N_STATE,
                    np: FHNModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::AdEx {
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_ref,
            v_peak,
            current_scale: _,
        } => {
            let pid = net.add_population::<AdExModel>(size);
            net.configure::<AdExModel>(pid, |p| {
                // AdEx layout (see AdExBuilder): dt, inv_tau_m, inv_tau_w,
                // v_rest, v_thresh, v_reset, delta_t, inv_delta_t, a, b,
                // tau_ref, v_peak; state v, w, refr.
                p.fill_param(0, dt_ms);
                p.fill_param(1, 1.0 / tau_m);
                p.fill_param(2, 1.0 / tau_w);
                p.fill_param(3, *v_rest);
                p.fill_param(4, *v_thresh);
                p.fill_param(5, *v_reset);
                p.fill_param(6, *delta_t);
                p.fill_param(7, 1.0 / delta_t);
                p.fill_param(8, *a);
                p.fill_param(9, *b);
                p.fill_param(10, *tau_ref);
                p.fill_param(11, *v_peak);
                p.fill_state(0, *v_rest);
                p.fill_state(1, 0.0);
                p.fill_state(2, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: ADEX_CUDA_SOURCE,
                    kernel_name: "adex_kernel",
                    ns: AdExModel::N_STATE,
                    np: AdExModel::N_PARAMS,
                    n: size,
                },
            )
        }
        // Классический Hodgkin–Huxley (hhCell): стандартные гейты m³h/n⁴.
        // q10 = 1 (без температурной шкалы), v_thresh = 0 (детекция спайка),
        // CaT выключен — ровно классический кальмаровый аксон.
        CellSpec::Hh {
            v0,
            g_na,
            e_na,
            g_k,
            e_k,
            g_leak,
            e_leak,
            inv_cm,
        } => {
            let pid = net.add_population::<HHModel>(size);
            net.configure::<HHModel>(pid, |p| {
                p.fill_param(HHModelParamIdx::dt as usize, dt_ms);
                p.fill_param(HHModelParamIdx::g_Na as usize, *g_na);
                p.fill_param(HHModelParamIdx::E_Na as usize, *e_na);
                p.fill_param(HHModelParamIdx::g_K as usize, *g_k);
                p.fill_param(HHModelParamIdx::E_K as usize, *e_k);
                p.fill_param(HHModelParamIdx::g_Leak as usize, *g_leak);
                p.fill_param(HHModelParamIdx::E_Leak as usize, *e_leak);
                p.fill_param(HHModelParamIdx::q10 as usize, 1.0);
                p.fill_param(HHModelParamIdx::v_thresh as usize, 0.0);
                p.fill_param(HHModelParamIdx::inv_cm as usize, *inv_cm);
                p.fill_param(HHModelParamIdx::g_CaT as usize, 0.0);
                p.fill_param(HHModelParamIdx::E_CaT as usize, 120.0);
                p.fill_state(HHModelStateIdx::v as usize, *v0);
                p.fill_state(HHModelStateIdx::m as usize, 0.05);
                p.fill_state(HHModelStateIdx::h as usize, 0.6);
                p.fill_state(HHModelStateIdx::n as usize, 0.32);
                p.fill_state(HHModelStateIdx::c as usize, 0.19);
                p.fill_state(HHModelStateIdx::was_below as usize, 1.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: HH_CUDA_SOURCE,
                    kernel_name: "hh_kernel",
                    ns: HHModel::N_STATE,
                    np: HHModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::ConductanceBased {
            v0,
            c_m,
            g_leak,
            e_leak,
            channels,
            markov_channels,
        } => {
            if markov_channels.is_empty() {
                // Произвольный HH-набор — SIMD generic-движок (не HHModel).
                let meta: Vec<ChannelMeta> = channels.iter().map(channel_to_meta).collect();
                let n_gates: usize = meta.iter().map(|m| m.n_gates).sum();
                let mut pop = ConductancePopulation::new(meta, size, dt_ms, celsius);
                pop.set_cm(*c_m);
                pop.set_leak(*g_leak, *e_leak);
                for (i, ch) in channels.iter().enumerate() {
                    pop.set_g_max(i, ch.g_max);
                }
                pop.set_resting_state(*v0);
                let pid = net.add_boxed_population(Box::new(pop));
                // Для data-driven каналов нет `#[dual_neuron]` CUDA-источника —
                // CPU-only популяция; пустой `source` честно помечает это.
                (
                    pid,
                    GpuPopMeta {
                        source: "",
                        kernel_name: "",
                        ns: 1 + n_gates,
                        np: 0,
                        n: size,
                    },
                )
            } else {
                // Есть кинетические (KS) каналы — скалярный master-equation
                // движок. HH-каналы опускаются до 2-состоянийного Маркова,
                // поэтому один `MarkovPopulation` гоняет всю смесь.
                let mut mc: Vec<MarkovChannel> = channels
                    .iter()
                    .map(|ch| MarkovChannel {
                        gates: ch
                            .gates
                            .iter()
                            .map(|(k, pow)| gate_kinetics_to_markov(k.clone(), *pow))
                            .collect(),
                        e_rev: ch.e_rev,
                        g_max: ch.g_max,
                    })
                    .collect();
                mc.extend(markov_channels.iter().map(|ch| MarkovChannel {
                    gates: ch.gates.clone(),
                    e_rev: ch.e_rev,
                    g_max: ch.g_max,
                }));
                let n_states: usize = mc.iter().flat_map(|c| &c.gates).map(|g| g.n_states()).sum();
                let mut pop = MarkovPopulation::new(mc, size, dt_ms, celsius);
                pop.set_cm(*c_m);
                pop.set_leak(*g_leak, *e_leak);
                // Установить g_max по порядку: сначала HH-каналы, затем KS.
                for (i, ch) in channels.iter().enumerate() {
                    pop.set_g_max(i, ch.g_max);
                }
                for (i, ch) in markov_channels.iter().enumerate() {
                    pop.set_g_max(channels.len() + i, ch.g_max);
                }
                pop.set_resting_state(*v0);
                let pid = net.add_boxed_population(Box::new(pop));
                (
                    pid,
                    GpuPopMeta {
                        source: "",
                        kernel_name: "",
                        ns: 1 + n_states,
                        np: 0,
                        n: size,
                    },
                )
            }
        }
    }
}
