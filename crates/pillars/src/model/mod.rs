pub mod lane;
pub mod context;
pub mod runner;
pub mod checker;
pub mod models;
pub mod sensors;
pub mod synapses;
pub mod util;
pub mod storage;
pub mod tensor_bridge;
#[cfg(feature = "rayon")]
pub mod orchestrator;

pub use lane::SimdLane;
pub use context::SimdContext;
pub use runner::run_neuron;
pub use runner::run_neuron_ext;
#[cfg(feature = "rayon")]
pub use runner::run_neuron_par;
pub use storage::StorageBuffer;
pub use tensor_bridge::TensorBridge;
#[cfg(feature = "rayon")]
pub use orchestrator::EngineOrchestrator;
pub use models::{
    LIFModel, IzhikevichModel, AdExModel, FHNModel, HAAPModel, HRModel, WCModel,
    MorrisLecarModel, GLIF3Model, GLIF4Model, HHModel, SAWModel,
    VSNModel, AstrocyteModel, CaPlasticityModel,
    AtlasovNeuromotorModel, Exp2SynModel,
};

/// Core Model Trait — Zero-Cost Hardware Abstraction for Neural Dynamics
pub trait Model {
    const N_PARAMS: usize;
    const N_STATE: usize;
    const HAS_SPIKE: bool;
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::checker::bio_lint;

    /// Run bio_lint for all 17 models — catches runtime bio-invariant violations.
    #[test]
    fn test_bio_lint_all_models() {
        // Spiking models
        bio_lint::<LIFModel>();
        bio_lint::<IzhikevichModel>();
        bio_lint::<AdExModel>();
        bio_lint::<HHModel>();
        bio_lint::<HAAPModel>();
        bio_lint::<FHNModel>();
        bio_lint::<HRModel>();
        bio_lint::<MorrisLecarModel>();
        bio_lint::<GLIF3Model>();
        bio_lint::<GLIF4Model>();
        bio_lint::<SAWModel>();
        bio_lint::<VSNModel>();
        // Non-spiking models
        bio_lint::<WCModel>();
        bio_lint::<AstrocyteModel>();
        bio_lint::<CaPlasticityModel>();
        bio_lint::<Exp2SynModel>();
        bio_lint::<AtlasovNeuromotorModel>();
    }

    #[test]
    fn test_lif_basic_dynamics() {
        let n = 8;
        let dt = 0.5;
        let tau_m = 10.0;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_ref = 2.0;

        let mut v = vec![v_rest; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 6];
        params[0..n].fill(dt);
        params[n..2*n].fill(1.0 / tau_m);
        params[2*n..3*n].fill(v_rest);
        params[3*n..4*n].fill(v_thresh);
        params[4*n..5*n].fill(v_reset);
        params[5*n..6*n].fill(tau_ref);

        unsafe {
            run_neuron::<LIFModel>(
                v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            );
        }

        assert!(v[0] > v_rest, "Membrane potential should depolarize: {}", v[0]);
    }

    #[test]
    fn test_lif_refractory_lockout() {
        let n = 8;
        let dt = 0.5;
        let tau_m = 10.0;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_ref = 2.0;

        let mut v = vec![v_rest; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_strong = 100.0;
        let i_syn = vec![i_strong; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 6];
        params[0..n].fill(dt);
        params[n..2*n].fill(1.0 / tau_m);
        params[2*n..3*n].fill(v_rest);
        params[3*n..4*n].fill(v_thresh);
        params[4*n..5*n].fill(v_reset);
        params[5*n..6*n].fill(tau_ref);

        let mut spike_steps = Vec::new();
        for step in 0..100 {
            unsafe {
                run_neuron::<LIFModel>(
                    v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                    i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
                );
            }
            if spikes[0] != 0 {
                spike_steps.push(step);
            }
        }

        assert!(spike_steps.len() >= 2, "Expected >=2 spikes with I={}, got {}", i_strong, spike_steps.len());

        for pair in spike_steps.windows(2) {
            let isi_ms = (pair[1] - pair[0]) as f32 * dt;
            assert!(isi_ms >= tau_ref - dt, "Refractory violation: ISI={}ms < tau_ref={}ms", isi_ms, tau_ref);
        }
    }

    #[test]
    fn test_lif_non_aligned() {
        let n = 10;
        let dt = 0.5;
        let tau_m = 10.0;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_ref = 2.0;
        let i_current = 30.0;

        let mut v = vec![v_rest; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![i_current; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 6];
        params[0..n].fill(dt);
        params[n..2*n].fill(1.0 / tau_m);
        params[2*n..3*n].fill(v_rest);
        params[3*n..4*n].fill(v_thresh);
        params[4*n..5*n].fill(v_reset);
        params[5*n..6*n].fill(tau_ref);

        for _ in 0..20 {
            unsafe {
                run_neuron::<LIFModel>(
                    v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                    i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
                );
            }
        }

        assert!((v[8] - v[0]).abs() < 1e-4, "Tail element v[8]={} != SIMD element v[0]={}", v[8], v[0]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_lif_parallel() {
        let n = 1024;
        let dt = 0.5;
        let tau_m = 10.0;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_ref = 2.0;

        let mut v = vec![v_rest; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 6];
        params[0..n].fill(dt);
        params[n..2*n].fill(1.0 / tau_m);
        params[2*n..3*n].fill(v_rest);
        params[3*n..4*n].fill(v_thresh);
        params[4*n..5*n].fill(v_reset);
        params[5*n..6*n].fill(tau_ref);

        unsafe {
            run_neuron_par::<LIFModel>(
                v.as_mut_ptr(),
                refr.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                i_syn.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
                256,
            );
        }

        assert_eq!(v.len(), 1024);
    }

    #[test]
    fn test_izhikevich_dynamics() {
        let n = 8;
        let dt = 0.5;
        let a = 0.02;
        let b = 0.2;
        let c = -65.0;
        let d = 8.0;
        let tau_ref = 2.0;
        let v_thresh = 30.0; // default threshold

        let mut v = vec![-65.0; n];
        let mut u = vec![b * -65.0; n];
        let mut refr = vec![0.0; n];
        let i_syn = vec![10.0; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * IzhikevichModel::N_PARAMS];
        params[0..n].fill(dt);
        params[n..2*n].fill(a);
        params[2*n..3*n].fill(b);
        params[3*n..4*n].fill(c);
        params[4*n..5*n].fill(d);
        params[5*n..6*n].fill(tau_ref);
        params[6*n..7*n].fill(v_thresh);

        unsafe {
            run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), refr.as_mut_ptr(),
                refr.as_mut_ptr(), refr.as_mut_ptr(), refr.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            );
        }

        assert!(v[0] > -65.0, "Izhikevich v should increase with input current");
    }

    #[test]
    fn test_fhn_dynamics() {
        let n = 8;
        let dt = 0.1;
        let tau_w = 12.5;
        let a = 0.7;
        let b = 0.8;
        let v_thresh = 0.5;

        let mut v = vec![-1.0; n];
        let mut w = vec![0.0; n];
        let mut was_below = vec![1.0; n];
        let i_syn = vec![1.5; n];
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 5];
        params[0..n].fill(dt);
        params[n..2*n].fill(1.0 / tau_w);
        params[2*n..3*n].fill(a);
        params[3*n..4*n].fill(b);
        params[4*n..5*n].fill(v_thresh);

        unsafe {
            run_neuron::<FHNModel>(
                v.as_mut_ptr(), w.as_mut_ptr(), was_below.as_mut_ptr(),
                was_below.as_mut_ptr(), was_below.as_mut_ptr(), was_below.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            );
        }

        assert!(v[0] > -1.0, "FHN v should increase with input current");
    }

    #[test]
    fn test_atlasov_neuromotor_dynamics() {
        let n = 8;
        let dt = 1.0;
        let d_ca = 0.5;
        let tau_pump = 20.0;
        let eta_reset = 0.8;
        let t_max = 100.0;
        let kd = 0.3;
        let l0 = 1.0;
        let vmax = 5.0;

        let mut ca = vec![0.0; n];
        let mut tension = vec![0.0; n];
        let mut reset_flag = vec![0.0; n];
        let mut length = vec![1.0; n];
        let mut velocity = vec![0.0; n];
        let i_syn = vec![1.0; n]; // Spike input
        let mut spikes = vec![0i32; n];

        let mut params = vec![0.0; n * 9];
        params[0..n].fill(dt);
        params[n..2*n].fill(d_ca);
        params[2*n..3*n].fill((-dt / tau_pump).exp());
        params[3*n..4*n].fill(eta_reset);
        params[4*n..5*n].fill(t_max);
        params[5*n..6*n].fill(kd);
        params[6*n..7*n].fill(l0);
        params[7*n..8*n].fill(1.0 / vmax);
        params[8*n..9*n].fill(1.0 / (0.5 * l0));

        unsafe {
            run_neuron::<AtlasovNeuromotorModel>(
                ca.as_mut_ptr(), tension.as_mut_ptr(), reset_flag.as_mut_ptr(),
                length.as_mut_ptr(), velocity.as_mut_ptr(), std::ptr::null_mut(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            );
        }

        assert!(ca[0] > 0.0, "Calcium concentration should rise on spike input: {}", ca[0]);
        assert!(tension[0] > 0.0, "Muscle tension should be generated: {}", tension[0]);
    }
}
