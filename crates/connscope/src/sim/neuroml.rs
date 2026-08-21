//! # NeuroML → CPU SIMD / GPU (`sim::neuroml`)
//!
//! Bridges [`pillars::api::neuroml`] (parse + compile a NeuroML2 document into
//! a [`pillars::api::Network`]) and [`pillars_cuda::network::CompiledNetwork`]
//! (the same network compiled onto the GPU) behind one small facade for the UI:
//!
//! ```text
//! .nml  ── parse ──► compile ──► Network (CPU SIMD)
//!                         └────► CompiledNetwork (CUDA, NVRTC)
//! ```
//!
//! * [`load`] parses + compiles the document and *also* builds the GPU mirror
//!   if a CUDA driver is present — if not, `gpu` is `None` with a warning and
//!   the CPU path still works.
//! * [`run_cpu`] / [`run_gpu`] run the same stimulus schedule (from the parsed
//!   `pulseGenerator`/`explicitInput` pairs) and return per-population spike
//!   counts, so CPU and GPU can be compared side by side.

use std::path::Path;

use pillars::api::neuroml::{self, LoadedNetwork};

/// Структурированная ошибка парсинга/компиляции NeuroML: несёт `pos`
/// (строка:колонка XML-ошибки, если известна) — редактор использует его,
/// чтобы подсветить строку, вместо разбора отформатированной строки.
pub use pillars::api::neuroml::NeuromlError;

/// GPU-зеркало сети. Без feature `cuda` типа (и поля `gpu`) просто нет —
/// приложение собирается без CUDA toolchain и работает только на CPU.
#[cfg(feature = "cuda")]
pub type GpuNetwork = pillars_cuda::network::CompiledNetwork;

/// Which engine runs the simulation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// pillars `Network` on CPU (AVX2 SIMD microkernels).
    Cpu,
    /// `CompiledNetwork` on a CUDA device (NVRTC + CUDA graph).
    Gpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU (SIMD)",
            Backend::Gpu => "GPU (CUDA)",
        }
    }

    /// Собран ли этот бэкенд в текущем бинарнике: GPU существует только
    /// с feature `cuda` (и требует драйвер/устройство — см. `NeuromlBundle::gpu`).
    pub fn available(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Gpu => cfg!(feature = "cuda"),
        }
    }
}

/// Per-population spike counts after a full run.
#[derive(Clone, Debug, Default)]
pub struct RunResult {
    /// Number of integration steps executed.
    pub steps: usize,
    /// Spike count per population (index = `PopulationId`).
    pub spike_counts: Vec<usize>,
    /// Sum of `spike_counts`.
    pub total_spikes: usize,
}

/// A loaded NeuroML model: the CPU network plus an optional compiled GPU mirror.
pub struct NeuromlBundle {
    pub loaded: LoadedNetwork,
    /// `None` when no CUDA driver/device was found (CPU still works).
    /// Существует только с feature `cuda`.
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuNetwork>,
    pub gpu_warning: Option<String>,
    pub n_pops: usize,
    pub n_neurons: usize,
    pub n_syn: usize,
    /// Neuron count per population (needed to split the flat GPU spike array).
    pub pop_sizes: Vec<usize>,
}

/// Read a `.nml` file, compile it and build the (optional) GPU mirror.
/// Uses [`pillars::api::neuroml::load_file`], so LEMS `<include href>`
/// resolves relative to the file's directory (multi-file models load).
pub fn load(path: &Path) -> Result<NeuromlBundle, String> {
    let loaded = neuroml::load_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bundle_from(loaded))
}

/// Parse + compile a NeuroML2 document (string) and build the GPU mirror.
///
/// В отличие от [`load`] возвращает **структурированную** [`NeuromlError`]
/// (с `pos` для XML-ошибок) — редактор/IDE используют её для подсветки строки.
pub fn load_str(xml: &str) -> Result<NeuromlBundle, NeuromlError> {
    let loaded = neuroml::load(xml)?;
    Ok(bundle_from(loaded))
}

/// Parse + compile a NeuroML2 document **CPU-only** — без GPU-зеркала и
/// per-population bookkeeping. Для живой валидации редактора на каждый кадр:
/// собирать GPU-зеркало на каждый чих расточительно (NVRTC-компиляция),
/// [`load_str`] это делает, а здесь — нет.
pub fn validate_str(xml: &str) -> Result<LoadedNetwork, NeuromlError> {
    neuroml::load(xml)
}

/// Build the [`NeuromlBundle`] (GPU mirror is best-effort) from a loaded network.
fn bundle_from(loaded: LoadedNetwork) -> NeuromlBundle {
    let pop_sizes: Vec<usize> = loaded.gpu_pops.iter().map(|m| m.n).collect();
    let n_neurons: usize = pop_sizes.iter().sum();
    let n_syn = loaded
        .network
        .graph()
        .map(|g| g.num_synapses())
        .unwrap_or(0);
    let n_pops = loaded.network.num_populations();
    // GPU is best-effort: no driver/device → `None` + a warning, CPU still runs.
    #[cfg(feature = "cuda")]
    let (gpu, gpu_warning) = match GpuNetwork::from_loaded(&loaded) {
        Ok(g) => (Some(g), None),
        Err(e) => (None, Some(e)),
    };
    // Без feature `cuda` GPU-бэкенда нет вовсе — это не ошибка, просто CPU-only.
    #[cfg(not(feature = "cuda"))]
    let gpu_warning: Option<String> = None;
    NeuromlBundle {
        loaded,
        #[cfg(feature = "cuda")]
        gpu,
        gpu_warning,
        n_pops,
        n_neurons,
        n_syn,
        pop_sizes,
    }
}

/// Run the full simulation on CPU SIMD (mirrors `LoadedNetwork::run`).
pub fn run_cpu(ln: &mut LoadedNetwork) -> RunResult {
    let r = ln.run();
    RunResult {
        steps: r.steps,
        total_spikes: r.spike_counts.iter().sum(),
        spike_counts: r.spike_counts,
    }
}

/// Per-compartment **V(t) traces** of one cable population, recorded while the
/// network steps (neuron 0 of the population; populations of morpho cells are
/// typically size 1). Decimated to at most `max_points` samples per trace.
#[derive(Clone, Debug)]
pub struct CableTraces {
    /// Sample times (ms).
    pub t_ms: Vec<f32>,
    /// `v[comp][sample]` (mV) — index 0 = root (soma) compartment.
    pub v: Vec<Vec<f32>>,
}

/// Step `ln` exactly like [`LoadedNetwork::run`] (stimulus schedule + spikes)
/// but additionally record per-compartment voltages of the cable population
/// `pop` into [`CableTraces`].
///
/// # Errors
/// If `pop` is not a cable population (see [`LoadedNetwork::cable_info`]).
pub fn run_traced(
    ln: &mut LoadedNetwork,
    pop: usize,
    max_points: usize,
) -> Result<CableTraces, String> {
    let info = ln
        .cable_info(pop)
        .ok_or_else(|| format!("population {pop} не кабельная (нет морфологии)"))?;
    let n_comps = info.compartments.len();
    let steps = (ln.duration_ms / ln.dt_ms).round().max(1.0) as usize;
    let stride = (steps / max_points.max(1)).max(1);
    let mut t_ms = Vec::new();
    let mut v = vec![Vec::new(); n_comps];
    for s in 0..steps {
        let t = s as f32 * ln.dt_ms;
        for (pop, indices, amps) in ln.injections_at(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        ln.network.step();
        if s % stride == 0 {
            if let Some(cv) = ln.network.cable_voltage(pop) {
                t_ms.push(t);
                for (c, tr) in v.iter_mut().enumerate() {
                    tr.push(cv.v[c]); // neuron 0
                }
            }
        }
    }
    Ok(CableTraces { t_ms, v })
}

/// Run the full simulation on GPU, mirroring [`LoadedNetwork::run`]: apply the
/// stimulus schedule, step, and count spikes per population from the downloaded
/// spike flags.
///
/// `pop_sizes` must match `loaded.gpu_pops[*].n` (returned by [`load`] /
/// [`load_str`] in [`NeuromlBundle::pop_sizes`]).
#[cfg(feature = "cuda")]
pub fn run_gpu(
    gpu: &mut GpuNetwork,
    ln: &LoadedNetwork,
    pop_sizes: &[usize],
) -> Result<RunResult, String> {
    let steps = (ln.duration_ms / ln.dt_ms).round().max(1.0) as usize;
    let mut spike_counts = vec![0usize; pop_sizes.len()];
    let mut spikes = vec![0i32; gpu.n_total()];
    for s in 0..steps {
        let t = s as f32 * ln.dt_ms;
        for (pop, indices, amps) in ln.injections_at(t) {
            gpu.inject_current(pop, &indices, &amps)?;
        }
        gpu.step()?;
        gpu.download_spikes(&mut spikes)?;
        let mut base = 0usize;
        for (p, &n) in pop_sizes.iter().enumerate() {
            for &sp in &spikes[base..base + n] {
                if sp != 0 {
                    spike_counts[p] += 1;
                }
            }
            base += n;
        }
    }
    let total_spikes = spike_counts.iter().sum();
    Ok(RunResult {
        steps,
        spike_counts,
        total_spikes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Тот же пример, что и в `resources/examples/neuroml/exc_inh_loop.nml`,
    /// зашит на этапе компиляции.
    const SAMPLE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/examples/neuroml/exc_inh_loop.nml"
    ));

    /// Морфо-клетка (сома + дендрит): `run_traced` записывает покомпартментные
    /// трассы — сома деполяризуется под импульсом сильнее кончика дендрита.
    #[test]
    fn run_traced_records_per_compartment_traces() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morpho_trace">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="20" z="0" diameter="2"/>
      <distal x="0" y="220" z="0" diameter="2"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
    <segmentGroup id="dendrite_group"><member segment="1"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific" segmentGroup="all"/>
      <initMembPotential value="-70mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <pulseGenerator id="pg" delay="0ms" duration="90ms" amplitude="0.1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;
        let mut ln = neuroml::load_with(xml, 0.05, 100.0).unwrap();
        let traces = run_traced(&mut ln, 0, 2000).unwrap();
        let n_comps = ln.cable_info(0).unwrap().compartments.len();
        assert_eq!(traces.v.len(), n_comps);
        assert_eq!(traces.t_ms.len(), 2000); // 2000 шагов, stride 1
        assert!(
            (traces.t_ms.last().unwrap() - 99.95).abs() < 1e-3,
            "last t = {}",
            traces.t_ms.last().unwrap()
        );
        for tr in &traces.v {
            assert_eq!(tr.len(), traces.t_ms.len());
        }
        // Сома деполяризуется под импульсом сильнее кончика дендрита.
        let soma_max = traces.v[0].iter().cloned().fold(f32::MIN, f32::max);
        let tip_max = traces.v[n_comps - 1]
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
        assert!(
            soma_max > tip_max + 1.0,
            "сома {soma_max} должна быть заметно выше кончика {tip_max}"
        );
    }

    /// CPU-прогон примера: RS-нейрон под импульсом спайкает, проекции
    /// распарсились (4 синапса), популяций ровно 2. GPU не трогаем —
    /// [`load_str`] деградирует до CPU, но здесь проверяем чистый CPU-путь.
    #[test]
    fn sample_cpu_run_produces_spikes() {
        let mut ln = neuroml::load(SAMPLE).expect("пример должен парситься");
        assert_eq!(ln.network.num_populations(), 2);
        assert_eq!(ln.network.graph().unwrap().num_synapses(), 4);
        let r = run_cpu(&mut ln);
        assert_eq!(r.steps, (ln.duration_ms / ln.dt_ms) as usize);
        assert_eq!(r.spike_counts.len(), 2);
        // exc[0] под импульсом I=10 стреляет (~22 Гц), популяция не пуста.
        assert!(r.spike_counts[0] > 0, "exc-популяция должна спайкать");
        assert!(r.total_spikes > 0);
    }

    /// Парсер отдаёт ошибку на мусорном XML (не паникует).
    #[test]
    fn load_str_rejects_garbage() {
        assert!(load_str("<neuroml").is_err());
    }
}
