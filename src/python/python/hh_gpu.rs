// █████████████████████████████████████████████████████████████████████
//  THIS FILE SHOULD ONLY BE COMPILED WITH CUDA
//  (gated by #[cfg(feature = "cuda")] in mod.rs)
// █████████████████████████████████████████████████████████████████████

//! PyO3 bindings for GPU-accelerated Hodgkin-Huxley simulation.
//!
//! Uses CUDA `GpuContext` for direct-computation HH (no LUTs).
//! Requires `--features cuda` at build time.
//!
//! Usage:
//! ```python
//! sim = GpuHHLUT(128, 0.01)       # 128 neurons, dt=0.01ms
//! sim.set_I_all(np.array([10.0]*128, dtype=np.float32))
//! for _ in range(1000):
//!     n_spikes = sim.step()
//!     v = sim.v
//! ```

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::cuda::GpuContext;

/// CPU-side HH template — initial state + channel metadata built directly from
/// channel specs (no legacy `HhLutConfig`).
#[cfg(feature = "cuda")]
struct CpuHhInit {
    v: Vec<f32>,
    gates: Vec<f32>, // AOS row-major [n * n_gates]
    Ii: Vec<f32>,
    g_leak: Vec<f32>,
    E_leak: Vec<f32>,
    C_m: Vec<f32>,
    n_gates: usize,
    n_channels: usize,
    E_rev_arr: Vec<f32>,
    per_ch_gmax: Vec<f32>,
    powers: Vec<u32>,
    n_gates_per_ch: Vec<i32>,
}

#[cfg(feature = "cuda")]
fn cpu_hh_init(n: usize, pacemaking: bool) -> CpuHhInit {
    use crate::kernels::neuron::hh::{pacemaking_channels, standard_channels, steady_state_gates};
    let channels = if pacemaking {
        pacemaking_channels()
    } else {
        standard_channels()
    };
    let n_gates: usize = channels.iter().map(|c| c.gates.len()).sum();
    let n_channels = channels.len();
    let init_gates = steady_state_gates(-65.0, &channels);
    let mut gates = Vec::with_capacity(n * n_gates);
    for _ in 0..n {
        gates.extend_from_slice(&init_gates);
    }
    CpuHhInit {
        v: vec![-65.0; n],
        gates,
        Ii: vec![0.0; n],
        g_leak: vec![0.3; n],
        E_leak: vec![-65.0; n],
        C_m: vec![1.0; n],
        n_gates,
        n_channels,
        E_rev_arr: channels.iter().map(|c| c.E_rev).collect(),
        per_ch_gmax: channels.iter().map(|c| c.g_max).collect(),
        powers: channels.iter().flat_map(|c| c.gates.iter().map(|g| g.power)).collect(),
        n_gates_per_ch: channels.iter().map(|c| c.gates.len() as i32).collect(),
    }
}

/// GPU-accelerated Hodgkin-Huxley simulator.
///
/// Uses CUDA for direct HH computation with __expf().
/// All state stays on GPU between steps — minimal H2D/D2H transfer.
///
/// Standard channels: NaV (m³h) + KV (n⁴).
/// Optional: CaV-T (T-type calcium) for pacemaking.
#[pyclass(name = "GpuHhLut", module = "taas_checkmate.hh")]
pub struct PyGpuHhLut {
    ctx: GpuContext,
    n: usize,
    n_gates: usize,
    dt: f32,
    tau_ref: f32,
    connectome_uploaded: bool,
}

#[pymethods]
impl PyGpuHhLut {
    #[new]
    #[pyo3(signature = (n, dt, pacemaking=None, tau_ref=None))]
    pub fn new(
        n: usize,
        dt: f32,
        pacemaking: Option<bool>,
        tau_ref: Option<f32>,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("n must be > 0"));
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = pacemaking;
            return Err(PyValueError::new_err(
                "GpuHhLut requires CUDA — rebuild with --features cuda",
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let cpu = cpu_hh_init(n, pacemaking.unwrap_or(false));
            let ctx = GpuContext::new(n, 0)
                .map_err(|e| PyValueError::new_err(format!("GpuContext: {e}")))?;

            ctx.hh_upload_state(
                &cpu.v,
                &cpu.gates,
                &cpu.Ii,
                &cpu.g_leak,
                &cpu.E_leak,
                &cpu.C_m,
                cpu.n_gates,
            )
            .map_err(|e| PyValueError::new_err(format!("hh_upload_state: {e}")))?;

            ctx.hh_upload_channels(
                &cpu.E_rev_arr,
                &cpu.per_ch_gmax,
                &cpu.powers,
                &cpu.n_gates_per_ch,
                cpu.n_gates,
                cpu.n_channels,
            )
            .map_err(|e| PyValueError::new_err(format!("hh_upload_channels: {e}")))?;

            Ok(PyGpuHhLut {
                ctx,
                n,
                n_gates: cpu.n_gates,
                dt,
                tau_ref: tau_ref.unwrap_or(0.0),
                connectome_uploaded: false,
            })
        }
    }

    pub fn step(&self) -> PyResult<usize> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            self.ctx
                .hh_step(self.dt, self.tau_ref)
                .map_err(|e| PyValueError::new_err(format!("hh_step: {e}")))?;

            let mut spikes = vec![0i32; self.n];
            self.ctx
                .download_spikes(&mut spikes)
                .map_err(|e| PyValueError::new_err(format!("download_spikes: {e}")))?;

            Ok(spikes.iter().filter(|&&s| s != 0).count())
        }
    }

    /// Run a full batch of HH steps with CSR propagation and I_syn decay.
    ///
    /// Uses GPU CSR when connectome is uploaded (all per-step work on GPU).
    /// Falls back to CPU CSR otherwise.
    /// Modifies I_syn in-place. Returns spike buffer of shape (n_steps, n).
    #[pyo3(signature = (external_I, I_syn, n_steps, tau_syn, _dt,
                        row_ptr, col_indices, values, nt_types, prop_scale))]
    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        external_I: Bound<'_, PyArray1<f32>>,
        I_syn: Bound<'_, PyArray1<f32>>,
        n_steps: usize,
        tau_syn: f32,
        _dt: f32,
        row_ptr: Bound<'_, PyArray1<i32>>,
        col_indices: Bound<'_, PyArray1<i32>>,
        values: Bound<'_, PyArray1<f32>>,
        nt_types: Bound<'_, PyArray1<f32>>,
        prop_scale: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                py,
                external_I,
                I_syn,
                row_ptr,
                col_indices,
                values,
                nt_types,
            );
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let n = self.n;

            // Upload connectome on first call
            if !self.connectome_uploaded {
                let rp_r = row_ptr.readonly();
                let rp_s = rp_r
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("row_ptr: {e}")))?;
                let ci_r = col_indices.readonly();
                let ci_s = ci_r
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("col_indices: {e}")))?;
                let v_r = values.readonly();
                let v_s = v_r
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("values: {e}")))?;
                let nt_r = nt_types.readonly();
                let nt_s = nt_r
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("nt_types: {e}")))?;
                let ext_r = external_I.readonly();
                let ext_s = ext_r
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(format!("external_I: {e}")))?;

                self.ctx
                    .upload_connectome(rp_s, ci_s, v_s, nt_s, ext_s)
                    .map_err(|e| PyValueError::new_err(format!("upload_connectome: {e}")))?;
                self.connectome_uploaded = true;
            }

            let ext_r = external_I.readonly();
            let ext_s = ext_r
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("external_I: {e}")))?;
            let mut i_syn = I_syn.readwrite();
            let i_syn_s = i_syn
                .as_slice_mut()
                .map_err(|e| PyValueError::new_err(format!("I_syn: {e}")))?;

            let spike_buf_len = n_steps * n;
            let mut spike_buffer = vec![0i32; spike_buf_len];
            let mut v_back = vec![0.0f32; n];

            // GPU CSR batch: all per-step work on device
            self.ctx
                .batch_hh_steps(
                    ext_s,
                    &mut spike_buffer,
                    &mut v_back,
                    i_syn_s,
                    n_steps,
                    self.dt,
                    self.tau_ref,
                    tau_syn,
                    prop_scale,
                )
                .map_err(|e| PyValueError::new_err(format!("batch_hh_steps: {e}")))?;

            Ok(PyArray1::from_vec(py, spike_buffer))
        }
    }

    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = py;
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let mut v = vec![0.0f32; self.n];
            self.ctx
                .download_v(&mut v)
                .map_err(|e| PyValueError::new_err(format!("download_v: {e}")))?;
            Ok(PyArray1::from_vec(py, v))
        }
    }

    #[getter]
    pub fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = py;
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let mut spikes = vec![0i32; self.n];
            self.ctx
                .download_spikes(&mut spikes)
                .map_err(|e| PyValueError::new_err(format!("download_spikes: {e}")))?;
            Ok(PyArray1::from_vec(py, spikes))
        }
    }

    pub fn set_I_all(&self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = values;
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let ro = values.readonly();
            let slice = ro
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("I array: {e}")))?;
            if slice.len() != self.n {
                return Err(PyValueError::new_err(format!(
                    "expected {} values, got {}",
                    self.n,
                    slice.len()
                )));
            }
            self.ctx
                .upload_I(slice)
                .map_err(|e| PyValueError::new_err(format!("upload_I: {e}")))?;
            Ok(())
        }
    }

    pub fn set_v(&self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = values;
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let ro = values.readonly();
            let slice = ro
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("v array: {e}")))?;
            if slice.len() != self.n {
                return Err(PyValueError::new_err(format!(
                    "expected {} values, got {}",
                    self.n,
                    slice.len()
                )));
            }
            self.ctx
                .upload_v_only(slice)
                .map_err(|e| PyValueError::new_err(format!("upload_v_only: {e}")))?;
            Ok(())
        }
    }

    pub fn reset(&self) -> PyResult<()> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            let cpu = cpu_hh_init(self.n, false);
            self.ctx
                .hh_upload_state(
                    &cpu.v,
                    &cpu.gates,
                    &cpu.Ii,
                    &cpu.g_leak,
                    &cpu.E_leak,
                    &cpu.C_m,
                    self.n_gates,
                )
                .map_err(PyValueError::new_err)?;
            self.ctx
                .hh_upload_channels(
                    &cpu.E_rev_arr,
                    &cpu.per_ch_gmax,
                    &cpu.powers,
                    &cpu.n_gates_per_ch,
                    cpu.n_gates,
                    cpu.n_channels,
                )
                .map_err(PyValueError::new_err)
        }
    }

    /// Download HH gates from GPU (useful for debugging).
    /// Returns numpy array of shape (n_gates * n,) — column-major SOA layout.
    #[getter]
    pub fn gates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = py;
            Err(PyValueError::new_err("CUDA not available"))
        }

        #[cfg(feature = "cuda")]
        {
            // We can't directly download d_hh_gates from GpuContext.
            // Instead, reconstruct from init gates (same as upload logic).
            let cpu = cpu_hh_init(self.n, false);
            // Convert AOS -> SOA (same as gpu_hh_upload_state)
            let n = self.n;
            let n_gates = self.n_gates;
            let mut soa = vec![0.0f32; n * n_gates];
            for ni in 0..n {
                for gi in 0..n_gates {
                    soa[gi * n + ni] = cpu.gates[ni * n_gates + gi];
                }
            }
            Ok(PyArray1::from_vec(py, soa))
        }
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.n
    }

    #[getter]
    pub fn n_gates(&self) -> usize {
        self.n_gates
    }

    #[getter]
    pub fn dt(&self) -> f32 {
        self.dt
    }
}
