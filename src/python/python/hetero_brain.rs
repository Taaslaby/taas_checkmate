//! Heterogeneous synaptic assemblies for the Python pillars facade.
//!
//! This module deliberately does not replace the legacy `ConnectomeBrain`.
//! `ConnectomeBrain` is a homogeneous LIF graph kept for compatibility;
//! `HeterogeneousConnectomeBrain` is the explicit multi-population API.

use numpy::{PyArray1, PyArrayMethods};
use pillars::prelude::*;
use pillars::zoo::{
    AdExModel, AtlasovNeuromotorModel, FHNModel, GLIF3Model, GLIF4Model, HAAPModel, HHModel,
    HRModel, IzhikevichDualModel, IzhikevichModel, LIFAtModel, LIFModel, MorrisLecarModel,
    RateModel, SAWModel, VSNModel, WCModel,
};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// A concrete population whose model remains fully statically dispatched.
enum Population {
    Lif(NeuralPopulation<LIFModel>),
    Izhikevich(NeuralPopulation<IzhikevichModel>),
    IzhikevichDual(NeuralPopulation<IzhikevichDualModel>),
    Hh(NeuralPopulation<HHModel>),
    Rate(NeuralPopulation<RateModel>),
    AdEx(NeuralPopulation<AdExModel>),
    Vsn(NeuralPopulation<VSNModel>),
    Fhn(NeuralPopulation<FHNModel>),
    Hr(NeuralPopulation<HRModel>),
    Glif3(NeuralPopulation<GLIF3Model>),
    Glif4(NeuralPopulation<GLIF4Model>),
    Haap(NeuralPopulation<HAAPModel>),
    MorrisLecar(NeuralPopulation<MorrisLecarModel>),
    Saw(NeuralPopulation<SAWModel>),
    Wc(NeuralPopulation<WCModel>),
    LifAt(NeuralPopulation<LIFAtModel>),
    Atlasov(NeuralPopulation<AtlasovNeuromotorModel>),
}

fn population_states_finite<M: Model>(pop: &NeuralPopulation<M>) -> bool {
    (0..M::N_STATE).all(|state| pop.state_slice(state).iter().all(|value| value.is_finite()))
}

impl Population {
    fn len(&self) -> usize {
        match self {
            Self::Lif(pop) => pop.len(),
            Self::Izhikevich(pop) => pop.len(),
            Self::IzhikevichDual(pop) => pop.len(),
            Self::Hh(pop) => pop.len(),
            Self::Rate(pop) => pop.len(),
            Self::AdEx(pop) => pop.len(),
            Self::Vsn(pop) => pop.len(),
            Self::Fhn(pop) => pop.len(),
            Self::Hr(pop) => pop.len(),
            Self::Glif3(pop) => pop.len(),
            Self::Glif4(pop) => pop.len(),
            Self::Haap(pop) => pop.len(),
            Self::MorrisLecar(pop) => pop.len(),
            Self::Saw(pop) => pop.len(),
            Self::Wc(pop) => pop.len(),
            Self::LifAt(pop) => pop.len(),
            Self::Atlasov(pop) => pop.len(),
        }
    }

    fn step(&mut self, input: &[f32], spikes: &mut [i32]) -> bool {
        match self {
            Self::Lif(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Izhikevich(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::IzhikevichDual(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Hh(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Rate(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::AdEx(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Vsn(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Fhn(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Hr(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Glif3(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Glif4(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Haap(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::MorrisLecar(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Saw(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Wc(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::LifAt(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
            Self::Atlasov(pop) => {
                pop.step(input, spikes);
                population_states_finite(pop)
            }
        }
    }

    /// Primary observable. For spiking models this is membrane voltage; for
    /// rate/field/muscle models it is their first canonical state column.
    fn primary_state(&self) -> &[f32] {
        match self {
            Self::Lif(pop) => pop.state_slice(0),
            Self::Izhikevich(pop) => pop.state_slice(0),
            Self::IzhikevichDual(pop) => pop.state_slice(0),
            Self::Hh(pop) => pop.state_slice(0),
            Self::Rate(pop) => pop.state_slice(0),
            Self::AdEx(pop) => pop.state_slice(0),
            Self::Vsn(pop) => pop.state_slice(0),
            Self::Fhn(pop) => pop.state_slice(0),
            Self::Hr(pop) => pop.state_slice(0),
            Self::Glif3(pop) => pop.state_slice(0),
            Self::Glif4(pop) => pop.state_slice(0),
            Self::Haap(pop) => pop.state_slice(0),
            Self::MorrisLecar(pop) => pop.state_slice(0),
            Self::Saw(pop) => pop.state_slice(0),
            Self::Wc(pop) => pop.state_slice(0),
            Self::LifAt(pop) => pop.state_slice(0),
            Self::Atlasov(pop) => pop.state_slice(0),
        }
    }
}

fn camel_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_lower_or_digit = false;
    for ch in s.chars() {
        if ch.is_ascii_uppercase() {
            if prev_lower_or_digit {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower_or_digit = false;
        } else {
            out.push(ch);
            prev_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

fn normalize_model(name: &str) -> String {
    let mut key = camel_to_snake(name.trim()).replace('-', "_");
    if let Some(stripped) = key.strip_suffix("population") {
        key = stripped.to_owned();
    }
    key.trim_end_matches('_').to_owned()
}

fn make_population(name: &str, n: usize) -> Result<Population, String> {
    if n == 0 {
        return Err("population size must be greater than zero".to_owned());
    }

    match normalize_model(name).as_str() {
        "lif" => {
            let mut pop = NeuralPopulation::<LIFModel>::new(n);
            pop.fill_param(0, 1.0);
            pop.fill_param(1, 0.1);
            pop.fill_param(2, -70.0);
            pop.fill_param(3, -54.0);
            pop.fill_param(4, -70.0);
            pop.fill_param(5, 2.0);
            pop.fill_state(0, -70.0);
            Ok(Population::Lif(pop))
        }
        "izhikevich" | "izh" => {
            let mut pop = NeuralPopulation::<IzhikevichModel>::new(n);
            pop.fill_param(0, 0.5);
            pop.fill_param(1, 0.02);
            pop.fill_param(2, 0.2);
            pop.fill_param(3, -65.0);
            pop.fill_param(4, 8.0);
            pop.fill_param(5, 2.0);
            pop.fill_param(6, 30.0);
            pop.fill_state(0, -65.0);
            pop.fill_state(1, -13.0);
            Ok(Population::Izhikevich(pop))
        }
        "izhikevich_dual" | "dual" => {
            let mut pop = NeuralPopulation::<IzhikevichDualModel>::new(n);
            for (idx, value) in [
                (0, 1.0),
                (1, 0.02),
                (2, 0.2),
                (3, -65.0),
                (4, 8.0),
                (5, 3.0),
                (6, 30.0),
                (7, 0.5),
                (8, 0.9),
                (9, 1.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            pop.fill_state(1, -13.0);
            Ok(Population::IzhikevichDual(pop))
        }
        "hh" => {
            let mut pop = NeuralPopulation::<HHModel>::new(n);
            for (idx, value) in [
                (0, 0.01),
                (1, 120.0),
                (2, 50.0),
                (3, 36.0),
                (4, -77.0),
                (5, 0.3),
                (6, -54.387),
                (7, 1.0),
                (8, 0.0),
                (9, 1.0),
                (10, 0.0),
                (11, 120.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            pop.fill_state(1, 0.05);
            pop.fill_state(2, 0.6);
            pop.fill_state(3, 0.32);
            pop.fill_state(4, 0.19);
            pop.fill_state(5, 1.0);
            Ok(Population::Hh(pop))
        }
        "rate" => {
            let mut pop = NeuralPopulation::<RateModel>::new(n);
            pop.fill_param(0, 0.001);
            pop.fill_param(1, 1.0);
            Ok(Population::Rate(pop))
        }
        "adex" | "ad_ex" => {
            let mut pop = NeuralPopulation::<AdExModel>::new(n);
            for (idx, value) in [
                (0, 0.5),
                (1, 1.0 / 20.0),
                (2, 1.0 / 100.0),
                (3, -65.0),
                (4, -50.0),
                (5, -70.0),
                (6, 2.0),
                (7, 0.5),
                (8, 0.0),
                (9, 2.0),
                (10, 2.0),
                (11, 20.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            Ok(Population::AdEx(pop))
        }
        "vsn" => {
            let mut pop = NeuralPopulation::<VSNModel>::new(n);
            for (idx, value) in [
                (0, 0.5),
                (1, 0.1),
                (2, -70.0),
                (3, -54.0),
                (4, -70.0),
                (5, 2.0),
                (6, 0.9),
                (7, 10.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -70.0);
            Ok(Population::Vsn(pop))
        }
        "fhn" => {
            let mut pop = NeuralPopulation::<FHNModel>::new(n);
            for (idx, value) in [(0, 0.01), (1, 0.1), (2, 0.7), (3, 0.8), (4, 0.5)] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(2, 1.0);
            Ok(Population::Fhn(pop))
        }
        "hr" => {
            let mut pop = NeuralPopulation::<HRModel>::new(n);
            for (idx, value) in [
                (0, 0.01),
                (1, 1.0),
                (2, 3.0),
                (3, 1.0),
                (4, 5.0),
                (5, 0.006),
                (6, 4.0),
                (7, -1.6),
                (8, 1.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -1.6);
            pop.fill_state(1, -12.0);
            Ok(Population::Hr(pop))
        }
        "glif3" => {
            let mut pop = NeuralPopulation::<GLIF3Model>::new(n);
            for (idx, value) in [
                (0, 1.0),
                (1, 0.1),
                (2, -65.0),
                (3, -50.0),
                (4, -65.0),
                (5, 0.02),
                (6, 0.01),
                (7, -80.0),
                (8, 2.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            Ok(Population::Glif3(pop))
        }
        "glif4" => {
            let mut pop = NeuralPopulation::<GLIF4Model>::new(n);
            for (idx, value) in [
                (0, 0.5),
                (1, 0.1),
                (2, -65.0),
                (3, -50.0),
                (4, -70.0),
                (5, 1.0 / 100.0),
                (6, 2.0),
                (7, -70.0),
                (8, 1.0 / 10.0),
                (9, 0.5),
                (10, 2.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            // `th` is an adaptive offset added to v_thresh_0; the neutral
            // initial condition is zero, not the absolute threshold.
            pop.fill_state(1, 0.0);
            Ok(Population::Glif4(pop))
        }
        "haap" => {
            let mut pop = NeuralPopulation::<HAAPModel>::new(n);
            for (idx, value) in [
                (0, 0.05),
                (1, 1.0),
                (2, 0.1),
                (3, -65.0),
                (4, 0.5),
                (5, -80.0),
                (6, 0.3),
                (7, -30.0),
                (8, 0.01),
                (9, 0.005),
                (10, -35.0),
                (11, 5.0),
                (12, -75.0),
                (13, 6.0),
                (14, -40.0),
                (15, -55.0),
                (16, 0.02),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            pop.fill_state(2, 1.0);
            Ok(Population::Haap(pop))
        }
        "morris_lecar" | "ml" => {
            let mut pop = NeuralPopulation::<MorrisLecarModel>::new(n);
            for (idx, value) in [
                (0, 1.0),
                (1, 0.05),
                (2, 2.0),
                (3, -60.0),
                (4, 4.4),
                (5, 120.0),
                (6, 8.0),
                (7, -84.0),
                (8, -1.2),
                (9, 18.0),
                (10, 2.0),
                (11, 30.0),
                (12, 0.04),
                (13, 0.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -60.0);
            pop.fill_state(2, 1.0);
            Ok(Population::MorrisLecar(pop))
        }
        "saw" => {
            let mut pop = NeuralPopulation::<SAWModel>::new(n);
            for (idx, value) in [
                (0, 1.0),
                (1, 0.02),
                (2, 0.2),
                (3, 0.01),
                (4, 0.0),
                (5, 2.0),
                (6, 30.0),
                (7, -65.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -65.0);
            Ok(Population::Saw(pop))
        }
        "wc" | "wilson_cowan" => {
            let mut pop = NeuralPopulation::<WCModel>::new(n);
            for (idx, value) in [
                (0, 0.1),
                (1, 0.1),
                (2, 0.1),
                (3, 1.2),
                (4, 0.4),
                (5, 0.6),
                (6, 0.8),
                (7, 4.0),
            ] {
                pop.fill_param(idx, value);
            }
            Ok(Population::Wc(pop))
        }
        "lifat" | "lif_at" => {
            let mut pop = NeuralPopulation::<LIFAtModel>::new(n);
            for (idx, value) in [
                (0, 0.5),
                (1, 0.1),
                (2, -70.0),
                (3, -54.0),
                (4, -70.0),
                (5, 0.05),
                (6, 5.0),
                (7, 3.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(0, -70.0);
            Ok(Population::LifAt(pop))
        }
        "atlasov_neuromotor" | "atlasov" => {
            let mut pop = NeuralPopulation::<AtlasovNeuromotorModel>::new(n);
            for (idx, value) in [
                (0, 1.0),
                (1, 0.1),
                (2, 0.9),
                (3, 0.5),
                (4, 1.0),
                (5, 0.2),
                (6, 1.0),
                (7, 1.0),
                (8, 1.0),
            ] {
                pop.fill_param(idx, value);
            }
            pop.fill_state(3, 1.0);
            Ok(Population::Atlasov(pop))
        }
        "triangulated" | "lut" => Err(
            "triangulated populations require TriangulatedPopulation.calibrate_lif() before assembly; use make_population after calibration or configure a LUT-aware brain"
                .to_owned(),
        ),
        other => Err(format!(
            "unsupported population model '{name}' (normalized '{other}'); heterogeneous graphs support all calibrated zoo models except LUT-only triangulated"
        )),
    }
}

/// A heterogeneous graph made of statically dispatched neuron populations.
///
/// This is intentionally a separate type from the legacy `ConnectomeBrain`,
/// which remains a homogeneous LIF-only compatibility class. The heterogeneous
/// backend supports all ordinary pillars zoo models with canonical defaults;
/// LUT-only `triangulated` is rejected until a calibration-aware brain API is
/// added.
///
/// `connectivity[post][pre]` is a block-level synaptic weight. A non-zero block
/// is all-to-all: every post neuron receives `weight * total_pre_spikes` on the
/// following step. Negative weights are valid and represent inhibition.
#[pyclass(
    name = "HeterogeneousConnectomeBrain",
    module = "taas_checkmate.pillars"
)]
pub struct PyHeterogeneousConnectomeBrain {
    populations: Vec<Population>,
    connectivity: Vec<Vec<f32>>, // [post][pre]
    currents: Vec<Vec<f32>>,     // delivered on the next step
    spikes: Vec<Vec<i32>>,
}

impl PyHeterogeneousConnectomeBrain {
    fn new(populations: Vec<Population>, connectivity: Vec<Vec<f32>>) -> Self {
        let currents = populations.iter().map(|pop| vec![0.0; pop.len()]).collect();
        let spikes = populations.iter().map(|pop| vec![0; pop.len()]).collect();
        Self {
            populations,
            connectivity,
            currents,
            spikes,
        }
    }

    fn append_spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for spikes in &self.spikes {
            result.append(PyArray1::from_slice(py, spikes))?;
        }
        Ok(result)
    }

    fn append_v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for pop in &self.populations {
            result.append(PyArray1::from_slice(py, pop.primary_state()))?;
        }
        Ok(result)
    }
}

#[pymethods]
impl PyHeterogeneousConnectomeBrain {
    /// Inject current into selected neurons of one population. The current is
    /// consumed by its next `step()` and is not silently broadcast.
    fn inject(&mut self, population: usize, indices: Vec<usize>, values: Vec<f32>) -> PyResult<()> {
        if population >= self.populations.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "population {population} out of range ({} populations)",
                self.populations.len()
            )));
        }
        if indices.len() != values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices and values must have the same length",
            ));
        }
        let current = &mut self.currents[population];
        for (&index, &value) in indices.iter().zip(values.iter()) {
            if index >= current.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "neuron index {index} out of range for population {population} (n={})",
                    current.len()
                )));
            }
            if !value.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "injected currents must be finite",
                ));
            }
            current[index] += value;
        }
        Ok(())
    }

    /// Add a complete per-neuron current vector to one population. Values are
    /// consumed by the next `step()` and must have exactly the population's length.
    fn inject_array(
        &mut self,
        population: usize,
        values: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        if population >= self.populations.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "population {population} out of range ({} populations)",
                self.populations.len()
            )));
        }
        let readonly = values.readonly();
        let src = readonly.as_slice()?;
        let current = &mut self.currents[population];
        if src.len() != current.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "inject_array: expected {} values for population {population}, got {}",
                current.len(),
                src.len()
            )));
        }
        if src.iter().any(|value| !value.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "injected currents must be finite",
            ));
        }
        for (dst, value) in current.iter_mut().zip(src.iter()) {
            *dst += *value;
        }
        Ok(())
    }

    /// Advance one step and return a Python list of spike arrays, one array per
    /// population. Synaptic currents generated here arrive next step.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        for index in 0..self.populations.len() {
            let finite =
                self.populations[index].step(&self.currents[index], &mut self.spikes[index]);
            if !finite {
                return Err(pyo3::exceptions::PyFloatingPointError::new_err(format!(
                    "population {index} produced a non-finite state"
                )));
            }
        }

        let mut next_currents: Vec<Vec<f32>> = self
            .populations
            .iter()
            .map(|pop| vec![0.0; pop.len()])
            .collect();
        for (post, row) in self.connectivity.iter().enumerate() {
            for (pre, &weight) in row.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                // Accumulate in i64 so a large but valid population cannot
                // overflow an i32 reduction before conversion to f32.
                let spike_count: i64 = self.spikes[pre].iter().map(|&spike| spike as i64).sum();
                let synaptic_current = weight * spike_count as f32;
                for current in &mut next_currents[post] {
                    *current += synaptic_current;
                }
            }
        }
        self.currents = next_currents;
        self.append_spikes(py)
    }

    #[getter]
    fn n_populations(&self) -> usize {
        self.populations.len()
    }

    #[getter]
    fn population_sizes(&self) -> Vec<usize> {
        self.populations.iter().map(Population::len).collect()
    }

    #[getter]
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.append_spikes(py)
    }

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.append_v(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "HeterogeneousConnectomeBrain(populations={}, sizes={:?})",
            self.populations.len(),
            self.population_sizes()
        )
    }
}

/// Build a heterogeneous all-to-all block-synaptic brain.
///
/// Returns `HeterogeneousConnectomeBrain`, not the legacy LIF-only
/// `ConnectomeBrain`. Existing `ConnectomeBrain(...)` callers are unchanged.
/// `populations` accepts every ordinary zoo model name used by
/// `make_population`, including class-style CamelCase names. `triangulated`
/// is intentionally rejected because it requires a process-wide LUT calibration
/// that cannot be represented by this two-argument constructor.
#[pyfunction]
#[pyo3(signature = (populations, connectivity))]
pub fn make_brain(
    py: Python<'_>,
    populations: Vec<(String, usize)>,
    connectivity: Vec<Vec<f32>>,
) -> PyResult<Py<PyHeterogeneousConnectomeBrain>> {
    let n_populations = populations.len();
    if n_populations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "populations must not be empty",
        ));
    }
    if connectivity.len() != n_populations {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "connectivity must have {n_populations} rows, got {}",
            connectivity.len()
        )));
    }
    for (row, values) in connectivity.iter().enumerate() {
        if values.len() != n_populations {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "connectivity row {row} must have {n_populations} columns, got {}",
                values.len()
            )));
        }
        if values.iter().any(|weight| !weight.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "connectivity row {row} contains NaN or infinity"
            )));
        }
    }

    let mut built = Vec::with_capacity(n_populations);
    for (name, n) in populations {
        built.push(make_population(&name, n).map_err(pyo3::exceptions::PyValueError::new_err)?);
    }
    Py::new(py, PyHeterogeneousConnectomeBrain::new(built, connectivity))
}
