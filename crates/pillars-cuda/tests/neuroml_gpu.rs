//! End-to-end: NeuroML2 XML → CPU SIMD **and** GPU, via the one-line bridge
//! [`CompiledNetwork::from_loaded`]. Proves the whole loop:
//!
//! ```text
//! <neuroml>  →  pillars::api::neuroml::load  →  LoadedNetwork
//!                     ├── .run()          → CPU SIMD (Network)
//!                     └── CompiledNetwork::from_loaded  → GPU (atomic scatter)
//! ```

use pillars::api::neuroml;
use pillars_cuda::CompiledNetwork;

/// Two heterogeneous populations (Izhikevich 2003 regular-spiking → Izhikevich
/// 2007 tonic), both driven by pulses, plus a delayed projection pre → post.
const XML: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="gputest">
  <izhikevichCell id="rs" v0="-65mV" thresh="30mV" a="0.02" b="0.2" c="-65mV" d="8"/>
  <izhikevich2007Cell id="tonic" v0="-60mV" C="100pF" k="0.7" vr="-60mV" vt="-40mV" vpeak="35mV" a="0.03" b="-2" c="-50mV" d="100"/>
  <pulseGenerator id="pg_pre" delay="0ms" duration="200ms" amplitude="10"/>
  <pulseGenerator id="pg_post" delay="0ms" duration="200ms" amplitude="100"/>
  <network id="net">
    <population id="pre" component="rs" size="4"/>
    <population id="post" component="tonic" size="4"/>
    <projection presynapticPopulation="pre" postsynapticPopulation="post">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="0.5" delay="1ms"/>
      <connection preCellId="../pre[1]" postCellId="../post[1]" weight="0.5" delay="1ms"/>
    </projection>
    <explicitInput target="pre[0]" input="pg_pre"/>
    <explicitInput target="pre[1]" input="pg_pre"/>
    <explicitInput target="post[0]" input="pg_post"/>
    <explicitInput target="post[1]" input="pg_post"/>
  </network>
</neuroml>
"#;

#[test]
fn neuroml_cpu_vs_gpu_parity() {
    // dt=1.0 ms: the Izhikevich-2007 tonic neuron is near-threshold sensitive
    // at dt=0.5 (1-ulp CPU/GPU drift flips spike counts there); dt=1.0 is the
    // proven-parity regime (see izhikevich2007_parity).
    let mut ln = neuroml::load_with(XML, 1.0, 200.0).expect("parse/compile the NeuroML doc");
    assert_eq!(ln.gpu_pops.len(), 2, "two populations recorded for GPU");

    // ── GPU (one-line bridge). Build it FIRST, while `ln` still holds the
    //       *initial* state — `ln.run()` below mutates the network in place. ──
    let mut cn = CompiledNetwork::from_loaded(&ln).expect("bridge to GPU (CUDA device required)");
    assert_eq!(cn.n_total(), ln.network.total_neurons());

    // ── CPU (from the same initial state) ─────────────────────────
    let cpu = ln.run();
    let n_total: usize = cpu.spike_counts.iter().sum();
    assert!(
        cpu.spike_counts[0] > 0 && cpu.spike_counts[1] > 0,
        "both populations must fire on CPU, got {:?}",
        cpu.spike_counts
    );

    // Per-population slice spans in the global spike array.
    let mut offs = 0usize;
    let pop_spans: Vec<(usize, usize)> = ln
        .gpu_pops
        .iter()
        .map(|m| {
            let span = (offs, offs + m.n);
            offs += m.n;
            span
        })
        .collect();

    let steps = (ln.duration_ms / ln.dt_ms).round().max(1.0) as usize;
    let mut gpu_counts = vec![0usize; ln.gpu_pops.len()];
    let mut spikes = vec![0i32; cn.n_total()];

    for s in 0..steps {
        let t = s as f32 * ln.dt_ms;
        for stim in &ln.inputs {
            if t >= stim.start_ms && t < stim.end_ms {
                let amps = vec![stim.amplitude; stim.indices.len()];
                cn.inject_current(stim.pop, &stim.indices, &amps).unwrap();
            }
        }
        cn.step().unwrap();
        cn.download_spikes(&mut spikes).unwrap();
        for (p, &(a, b)) in pop_spans.iter().enumerate() {
            gpu_counts[p] += spikes[a..b].iter().sum::<i32>() as usize;
        }
    }

    for p in 0..ln.gpu_pops.len() {
        assert_eq!(
            cpu.spike_counts[p], gpu_counts[p],
            "population {p} spike count diverged: CPU={} GPU={}",
            cpu.spike_counts[p], gpu_counts[p]
        );
    }
    eprintln!(
        "[parity] neuroml_gpu: CPU={:?} GPU={:?} total={n_total} over {steps} steps",
        cpu.spike_counts, gpu_counts
    );
}
