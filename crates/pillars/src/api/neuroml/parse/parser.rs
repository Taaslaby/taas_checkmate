// ─── Parser ────────────────────────────────────────────────────
use super::*;

use std::collections::{HashMap, HashSet};

/// Parse a NeuroML2 (and/or LEMS) XML document into the IR.
pub(crate) fn parse(xml: &str) -> Result<Ir, NeuromlError> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| {
        // Позиция XML-ошибки (row:col, 1-based) — структурированно, для
        // редактора/IDE: без неё пользователь ищет синтаксическую ошибку по
        // всему файлу. `Display` сам собирает «XML: R:C: …» из `pos`.
        let pos = e.pos();
        if pos.row == 1 && pos.col == 1 {
            // roxmltree не смог локализовать ошибку (напр. пустой документ) —
            // оставляем пометку «XML:», но без позиции.
            NeuromlError::parse(format!("XML: {e}"))
        } else {
            NeuromlError::xml(pos.row, pos.col, e.to_string())
        }
    })?;


    // Pre-passes (see `prepass`): order-independent resolution of LEMS
    // ComponentTypes, ion channels, morphologies and biophysics.
    let (comp_types, pool_type_names) = super::prepass::collect_component_types(&doc)?;
    let (channels, ks_channels, morphologies, biophysics, concentration_models) =
        super::prepass::collect_channels_morphologies(&doc, &comp_types, &pool_type_names)?;

    let mut ir = Ir {
        cells: Vec::new(),
        synapses: Vec::new(),
        double_synapses: Vec::new(),
        pulse_gens: Vec::new(),
        poisson_synapses: Vec::new(),
        transient_poisson_synapses: Vec::new(),
        spike_gens: Vec::new(),
        spike_arrays: Vec::new(),
        spike_array_refs: Vec::new(),
        sine_gens: Vec::new(),
        ramp_gens: Vec::new(),
        compounds: Vec::new(),
        timed_syn_inputs: Vec::new(),
        voltage_clamps: Vec::new(),
        voltage_clamp_triples: Vec::new(),
        pops: Vec::new(),
        projections: Vec::new(),
        inputs: Vec::new(),
        gap_junctions: Vec::new(),
        electrical_projections: Vec::new(),
        graded_synapses: Vec::new(),
        sigmoid_graded_synapses: Vec::new(),
        continuous_projections: Vec::new(),
        channels: Vec::new(),
        ks_channels: Vec::new(),
        morphologies: Vec::new(),
        biophysics: HashMap::new(),
        concentration_models: HashMap::new(),
        morpho_cells: Vec::new(),
        sim: None,
        warnings: HashSet::new(),
    };

    for node in doc.root_element().descendants() {
        match node.tag_name().name() {
            // ── cells ──
            "izhikevichCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Izh {
                        v0: voltage_attr(node, "v0")?,
                        thresh: voltage_attr(node, "thresh")?,
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        c: voltage_attr(node, "c")?,
                        d: raw_attr(node, "d")?,
                    },
                };
                ir.cells.push(cell);
            }
            "izhikevich2007Cell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Izh2007 {
                        v0: voltage_attr(node, "v0")?,
                        cap: raw_attr(node, "C")?,
                        k: raw_attr(node, "k")?,
                        vr: voltage_attr(node, "vr")?,
                        vt: voltage_attr(node, "vt")?,
                        vpeak: voltage_attr(node, "vpeak")?,
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        c: voltage_attr(node, "c")?,
                        d: raw_attr(node, "d")?,
                    },
                };
                ir.cells.push(cell);
            }
            "iafTauRefCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: time_attr(node, "tau")?,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref: time_attr(node, "refract")?,
                        // iafTau*Cell: ток уже в mV-эквиваленте (нет gL).
                        current_scale: 1.0,
                    },
                };
                ir.cells.push(cell);
            }
            "iafTauCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: time_attr(node, "tau")?,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref: 0.0,
                        current_scale: 1.0,
                    },
                };
                ir.cells.push(cell);
            }
            "iafCell" | "iafRefCell" => {
                // tau_m = C / gL — raw pF / nS gives ms directly.
                let c = raw_attr(node, "C")?;
                let gl = raw_attr(node, "leakConductance")?;
                if gl <= 0.0 {
                    return Err(NeuromlError::parse(
                        "iafCell: leakConductance должно быть > 0",
                    ));
                }
                // iafRefCell добавляет refract; у iafCell его нет (0).
                let tau_ref = attr_opt(node, "refract")
                    .map(to_ms)
                    .transpose()?
                    .unwrap_or(0.0);
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: c / gl,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref,
                        // LIF voltage-normalised: ток nA→мВ = ·1000/gL.
                        current_scale: 1000.0 / gl,
                    },
                };
                ir.cells.push(cell);
            }
            "adExIaFCell" => {
                let c = raw_attr(node, "C")?;
                let gl = raw_attr(node, "gL")?;
                if gl <= 0.0 {
                    return Err(NeuromlError::parse("adExIaFCell: gL должно быть > 0"));
                }
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::AdEx {
                        tau_m: c / gl,
                        tau_w: time_attr(node, "tauw")?,
                        v_rest: voltage_attr(node, "EL")?,
                        // VT = exponential spike threshold (pillars v_thresh),
                        // thresh = spike *detection* threshold (pillars v_peak).
                        v_thresh: voltage_attr(node, "VT")?,
                        v_reset: voltage_attr(node, "reset")?,
                        delta_t: voltage_attr(node, "delT")?,
                        // Pillars AdEx — voltage-normalised (÷gL). nA→mV:
                        // 1 nA / 1 nS = 1 V = 1000 mV, так что b/gL·1000.
                        a: raw_attr(node, "a")? / gl,
                        b: raw_attr(node, "b")? / gl * 1000.0,
                        tau_ref: time_attr(node, "refract")?,
                        v_peak: voltage_attr(node, "thresh")?,
                        current_scale: 1000.0 / gl,
                    },
                };
                ir.cells.push(cell);
            }
            // Классический FitzHugh–Nagumo (безразмерный, единственный
            // параметр — встроенный ток I):
            //   dV/dt = V − V³/3 − W + I ; dW/dt = 0.08·(V + 0.7 − 0.8·W)
            // → pillars FHNModel с a=0.7, b=0.8, inv_tau_w=0.08.
            "fitzHughNagumoCell" => {
                ir.cells.push(Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Fhn {
                        a: 0.7,
                        b: 0.8,
                        inv_tau_w: 0.08,
                        v_thresh: 0.5,
                        bias: raw_attr(node, "I")?,
                        v0: 0.0,
                        w0: 0.0,
                    },
                });
            }
            // FitzHugh–Nagumo 1969: явные a, b, phi (масштаб времени) и I.
            //   dV/dt = V − V³/3 − W + I ; dW/dt = φ·(V + a − b·W)
            "fitzHughNagumo1969Cell" => {
                ir.cells.push(Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Fhn {
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        inv_tau_w: raw_attr(node, "phi")?,
                        v_thresh: 0.5,
                        bias: raw_attr(node, "I")?,
                        v0: raw_attr_opt(node, "V0")?.unwrap_or(0.0),
                        w0: raw_attr_opt(node, "W0")?.unwrap_or(0.0),
                    },
                });
            }
            // Классический Hodgkin–Huxley через channelDensity + leak. Гейты
            // резолвятся по объявленным `ionChannelHH` (data-driven), с
            // фолбэком на подстроки id/ionChannel для легаси-моделей.
            // Стандартные m³h/n⁴ → быстрый путь HHModel; иначе — generic-движок.
            "hhCell" => {
                let v0 = voltage_attr(node, "v0")?;
                let c = raw_attr(node, "C")?;
                let g_leak = raw_attr(node, "leakConductance")?;
                let e_leak = voltage_attr(node, "leakReversal")?;
                if c <= 0.0 {
                    return Err(NeuromlError::parse("hhCell: C должно быть > 0"));
                }
                // Na/K — для быстрого пути HHModel, если каналы стандартные.
                let mut g_na: Option<(f32, f32)> = None; // (condDensity, erev)
                let mut g_k: Option<(f32, f32)> = None;
                // Нестандартные каналы — для generic-движка.
                let mut custom: Vec<ResolvedChannel> = Vec::new();
                let mut markov_custom: Vec<ResolvedMarkovChannel> = Vec::new();

                for child in node.children() {
                    if child.tag_name().name() != "channelDensity" {
                        continue;
                    }
                    let cond = raw_attr(child, "condDensity")?;
                    let erev = voltage_attr(child, "erev")?;
                    let ion = attr(child, "ionChannel")?;
                    if let Some(ch) = channels.get(ion) {
                        if !ch.fractional_gates.is_empty() {
                            // Есть gateFractional — SIMD HH-движок не умеет
                            // взвешенную сумму субгейтов; опускаем весь канал до
                            // марковской схемы (HH → 2-состояния, fractional →
                            // произведение субгейтов).
                            markov_custom.push(resolve_markov_channel(
                                ch,
                                erev,
                                cond,
                                CONDUCTANCE_CELSIUS,
                            )?);
                        } else {
                            match standard_role(ch) {
                                Some(StandardChannel::Nav) => g_na = Some((cond, erev)),
                                Some(StandardChannel::Kv) => g_k = Some((cond, erev)),
                                None => custom.push(resolve_channel(
                                    ch,
                                    erev,
                                    cond,
                                    CONDUCTANCE_CELSIUS,
                                )?),
                            }
                        }
                    } else if let Some(ch) = ks_channels.get(ion) {
                        markov_custom.push(ResolvedMarkovChannel {
                            gates: ch.gates.clone(),
                            // ionChannelKS gates carry no q10Settings.
                            q10s: vec![1.0; ch.gates.len()],
                            e_rev: erev,
                            g_max: cond,
                        });
                    } else {
                        // Нет объявления — легаси-эвристика по подстрокам id/ionChannel.
                        let id = attr(child, "id")?.to_lowercase();
                        let ion_l = ion.to_lowercase();
                        if id.contains("na") || ion_l.contains("na") {
                            g_na = Some((cond, erev));
                        } else if id.contains("k") || ion_l.contains("k") {
                            g_k = Some((cond, erev));
                        }
                    }
                }

                if custom.is_empty() && markov_custom.is_empty() {
                    // Только стандартный Na/K — быстрый путь HHModel.
                    let (g_na, e_na) = g_na.ok_or_else(|| {
                        NeuromlError::parse(
                            "hhCell: не найден Na-канал (channelDensity с id/ionChannel, содержащим «na»)",
                        )
                    })?;
                    let (g_k, e_k) = g_k.ok_or_else(|| {
                        NeuromlError::parse(
                            "hhCell: не найден K-канал (channelDensity с id/ionChannel, содержащим «k»)",
                        )
                    })?;
                    ir.cells.push(Cell {
                        id: attr(node, "id")?.to_string(),
                        spec: CellSpec::Hh {
                            v0,
                            g_na,
                            e_na,
                            g_k,
                            e_k,
                            g_leak,
                            e_leak,
                            inv_cm: 1.0 / c,
                        },
                    });
                } else {
                    // Смешанный/нестандартный набор — generic-движок. Порядок:
                    // стандартные Na/K (если есть), затем нестандартные по
                    // объявлению. Кинетические (KS) каналы — отдельным списком:
                    // если их нет, работает SIMD-движок, иначе скалярный Markov.
                    let mut all: Vec<ResolvedChannel> = Vec::new();
                    if let Some((cond, erev)) = g_na {
                        all.push(standard_resolved(StandardChannel::Nav, erev, cond));
                    }
                    if let Some((cond, erev)) = g_k {
                        all.push(standard_resolved(StandardChannel::Kv, erev, cond));
                    }
                    all.extend(custom);
                    ir.cells.push(Cell {
                        id: attr(node, "id")?.to_string(),
                        spec: CellSpec::ConductanceBased {
                            v0,
                            c_m: c,
                            g_leak,
                            e_leak,
                            channels: all,
                            markov_channels: markov_custom,
                        },
                    });
                }
            }
            // ── synapses ──
            "expOneSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpOne {
                        tau_decay: attr_opt(node, "tauDecay").map(to_ms).transpose()?,
                    },
                    plast: None,
                });
            }
            "expTwoSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpTwo {
                        tau_rise: attr_opt(node, "tauRise").map(to_ms).transpose()?,
                        tau_decay: attr_opt(node, "tauDecay").map(to_ms).transpose()?,
                    },
                    plast: None,
                });
            }
            "alphaSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::Alpha {
                        tau: to_ms(attr(node, "tau")?)?,
                    },
                    plast: None,
                });
            }
            "expThreeSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: None,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpThree {
                        tau_rise: to_ms(attr(node, "tauRise")?)?,
                        tau_decay1: to_ms(attr(node, "tauDecay1")?)?,
                        tau_decay2: to_ms(attr(node, "tauDecay2")?)?,
                        gbase1: raw_attr(node, "gbase1")?,
                        gbase2: raw_attr(node, "gbase2")?,
                    },
                    plast: None,
                });
            }
            "doubleSynapse" => {
                ir.double_synapses.push(DoubleSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    synapse1: attr(node, "synapse1")?.to_string(),
                    synapse2: attr(node, "synapse2")?.to_string(),
                });
            }
            "blockingPlasticSynapse" => {
                let mut plast = None;
                for child in node.children() {
                    match child.tag_name().name() {
                        "tsodyksMarkramDepMechanism" => {
                            plast = Some(PlasticitySpec::TsodyksDep {
                                init_release_prob: raw_attr(child, "initReleaseProb")?,
                                tau_rec: to_ms(attr(child, "tauRec")?)?,
                            });
                        }
                        "tsodyksMarkramDepFacMechanism" => {
                            plast = Some(PlasticitySpec::TsodyksDepFac {
                                init_release_prob: raw_attr(child, "initReleaseProb")?,
                                tau_rec: to_ms(attr(child, "tauRec")?)?,
                                tau_fac: to_ms(attr(child, "tauFac")?)?,
                            });
                        }
                        _ => {}
                    }
                }
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpTwo {
                        tau_rise: Some(to_ms(attr(node, "tauRise")?)?),
                        tau_decay: Some(to_ms(attr(node, "tauDecay")?)?),
                    },
                    plast,
                });
            }
            // ── input ──
            "pulseGenerator" => {
                ir.pulse_gens.push(PulseGen {
                    id: attr(node, "id")?.to_string(),
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    amplitude: raw_attr(node, "amplitude")?,
                });
            }
            "poissonFiringSynapse" => {
                ir.poisson_synapses.push(PoissonSpec {
                    id: attr(node, "id")?.to_string(),
                    rate_hz: to_hz(attr(node, "averageRate")?)?,
                });
            }
            "transientPoissonFiringSynapse" => {
                ir.transient_poisson_synapses.push(TransientPoissonSpec {
                    id: attr(node, "id")?.to_string(),
                    rate_hz: to_hz(attr(node, "averageRate")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                });
            }
            "spikeGenerator" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Regular {
                        period_ms: time_attr(node, "period")?,
                    },
                });
            }
            "spikeGeneratorPoisson" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Poisson {
                        rate_hz: to_hz(attr(node, "averageRate")?)?,
                    },
                });
            }
            "spikeGeneratorRandom" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Random {
                        min_isi_ms: time_attr(node, "minISI")?,
                        max_isi_ms: time_attr(node, "maxISI")?,
                    },
                });
            }
            "spikeGeneratorRefPoisson" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::RefPoisson {
                        rate_hz: to_hz(attr(node, "averageRate")?)?,
                        min_isi_ms: time_attr(node, "minimumISI")?,
                    },
                });
            }
            "spikeArray" => {
                let mut times = Vec::new();
                for sp in node.children() {
                    if sp.is_element() && sp.tag_name().name() == "spike" {
                        times.push(to_ms(attr(sp, "time")?)?);
                    }
                }
                times.sort_by(f32::total_cmp);
                ir.spike_arrays.push(SpikeArray {
                    id: attr(node, "id")?.to_string(),
                    times_ms: times,
                });
            }
            "spikeArrayRef" => {
                ir.spike_array_refs.push(SpikeArrayRef {
                    id: attr(node, "id")?.to_string(),
                    array: attr(node, "ref")?.to_string(),
                });
            }
            "sineGenerator" => {
                ir.sine_gens.push(SineGenSpec {
                    id: attr(node, "id")?.to_string(),
                    phase: raw_attr(node, "phase")?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    amplitude: raw_attr(node, "amplitude")?,
                    period_ms: time_attr(node, "period")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "rampGenerator" => {
                ir.ramp_gens.push(RampGenSpec {
                    id: attr(node, "id")?.to_string(),
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    start_amp: raw_attr(node, "startAmplitude")?,
                    finish_amp: raw_attr(node, "finishAmplitude")?,
                    baseline_amp: raw_attr(node, "baselineAmplitude")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "compoundInput" => {
                let mut children = Vec::new();
                for child in node.children() {
                    if child.is_element() && child.tag_name().name() == "input" {
                        children.push(attr(child, "component")?.to_string());
                    }
                }
                ir.compounds.push(CompoundSpec {
                    id: attr(node, "id")?.to_string(),
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                    children,
                });
            }
            "timedSynapticInput" => {
                let mut times = Vec::new();
                for sp in node.children() {
                    if sp.is_element() && sp.tag_name().name() == "spike" {
                        times.push(to_ms(attr(sp, "time")?)?);
                    }
                }
                times.sort_by(f32::total_cmp);
                ir.timed_syn_inputs.push(TimedSynInputSpec {
                    id: attr(node, "id")?.to_string(),
                    synapse: attr(node, "synapse")?.to_string(),
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                    times_ms: times,
                });
            }
            "voltageClamp" => {
                ir.voltage_clamps.push(VoltageClampSpec {
                    id: attr(node, "id")?.to_string(),
                    target_mv: voltage_attr(node, "targetVoltage")?,
                    series_resistance_mohm: to_mohm(attr(node, "simpleSeriesResistance")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "voltageClampTriple" => {
                ir.voltage_clamp_triples.push(VoltageClampTripleSpec {
                    id: attr(node, "id")?.to_string(),
                    active: raw_attr(node, "active")? != 0.0,
                    conditioning_mv: voltage_attr(node, "conditioningVoltage")?,
                    testing_mv: voltage_attr(node, "testingVoltage")?,
                    return_mv: voltage_attr(node, "returnVoltage")?,
                    series_resistance_mohm: to_mohm(attr(node, "simpleSeriesResistance")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            // ── structure ──
            "population" => {
                let size: usize = raw_attr(node, "size")? as usize;
                ir.pops.push(PopSpec {
                    id: attr(node, "id")?.to_string(),
                    component: attr(node, "component")?.to_string(),
                    size,
                });
            }
            "projection" => {
                let mut proj = ProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    synapse: attr_opt(node, "synapse").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "connection" | "connectionWD" => {
                            let (_, pre_idx) = parse_cell_ref(attr(child, "preCellId")?)?;
                            let (_, post_idx) = parse_cell_ref(attr(child, "postCellId")?)?;
                            proj.connections.push(Connection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                                delay_ms: attr_opt(child, "delay")
                                    .map(to_ms)
                                    .transpose()?
                                    .unwrap_or(0.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.projections.push(proj);
            }
            "explicitInput" => {
                let (pop, idx) = parse_cell_ref(attr(node, "target")?)?;
                ir.inputs.push(InputSpec {
                    target_pop: pop.to_string(),
                    target_idx: idx,
                    generator: attr(node, "input")?.to_string(),
                });
            }
            "gapJunction" => {
                ir.gap_junctions.push(GapJunctionSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr_opt(node, "conductance")?,
                });
            }
            "electricalProjection" => {
                let mut proj = ElectricalProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    gap_junction: attr_opt(node, "gapJunction").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "electricalConnection"
                        | "electricalConnectionInstance"
                        | "electricalConnectionInstanceW" => {
                            // id-form spells the refs `preCellId`/`postCellId`,
                            // instance-form spells them `preCell`/`postCell`.
                            let pre = attr_opt(child, "preCellId")
                                .or_else(|| attr_opt(child, "preCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing preCellId/preCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let post = attr_opt(child, "postCellId")
                                .or_else(|| attr_opt(child, "postCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing postCellId/postCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let (_, pre_idx) = parse_cell_ref(pre)?;
                            let (_, post_idx) = parse_cell_ref(post)?;
                            proj.connections.push(ElectricalConnection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.electrical_projections.push(proj);
            }
            "linearGradedSynapse" => {
                ir.graded_synapses.push(GradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr_opt(node, "conductance")?,
                });
            }
            "silentSynapse" => {
                ir.graded_synapses.push(GradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: Some(0.0),
                });
            }
            "gradedSynapse" => {
                ir.sigmoid_graded_synapses.push(SigmoidGradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr(node, "conductance")?,
                    delta: voltage_attr(node, "delta")?,
                    k: to_per_ms(attr(node, "k")?)?,
                    vth: voltage_attr(node, "Vth")?,
                    erev: voltage_attr(node, "erev")?,
                });
            }
            "continuousProjection" => {
                let mut proj = ContinuousProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    synapse: attr_opt(node, "synapse").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "continuousConnection" | "continuousConnectionInstance" => {
                            let pre = attr_opt(child, "preCellId")
                                .or_else(|| attr_opt(child, "preCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing preCellId/preCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let post = attr_opt(child, "postCellId")
                                .or_else(|| attr_opt(child, "postCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing postCellId/postCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let (_, pre_idx) = parse_cell_ref(pre)?;
                            let (_, post_idx) = parse_cell_ref(post)?;
                            proj.connections.push(ContinuousConnection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.continuous_projections.push(proj);
            }
            // ── LEMS simulation ──
            "Simulation" => {
                ir.sim = Some(SimConfig {
                    dt_ms: time_attr(node, "step")?,
                    duration_ms: time_attr(node, "length")?,
                });
            }
            // ── morphologically-detailed cell ──
            "cell" => {
                // Reference form: `<cell id="…" morphology="…" biophysicalProperties="…"/>`.
                // Inline form (also valid NeuroML2):
                // `<cell id="…"><morphology id="…">…</morphology><biophysicalProperties id="…">…</biophysicalProperties></cell>`
                // — the pre-pass already registered the inline elements, so we
                // just resolve the id from the inline child when the attribute
                // is absent.
                let id = attr(node, "id")?.to_string();
                let morphology = match attr_opt(node, "morphology") {
                    Some(m) => m.to_string(),
                    None => node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "morphology")
                        .map(|m| attr(m, "id").map(str::to_string))
                        .transpose()?
                        .ok_or_else(|| {
                            NeuromlError::parse(format!(
                                "<cell id=\"{id}\">: missing `morphology` (attribute or inline child)"
                            ))
                        })?,
                };
                let biophysics = match attr_opt(node, "biophysicalProperties") {
                    Some(b) => Some(b.to_string()),
                    None => node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "biophysicalProperties")
                        .map(|b| attr(b, "id").map(str::to_string))
                        .transpose()?,
                };
                ir.morpho_cells.push(MorphoCell {
                    id,
                    morphology,
                    biophysics,
                });
            }
            // Распознанные, но неподдерживаемые элементы — предупреждаем,
            // а не молчим (модель «загрузилась», но потеряла связность/вход).
            other => {
                if let Some(note) = skipped_element_note(other) {
                    let msg = format!("<{other}> — не поддерживается, пропущено{note}");
                    ir.warnings.insert(msg);
                }
            }
        }
    }

    ir.channels = channels.into_values().collect();
    ir.ks_channels = ks_channels.into_values().collect();
    ir.morphologies = morphologies;
    ir.biophysics = biophysics;
    ir.concentration_models = concentration_models;
    Ok(ir)
}

/// Short note for recognised-but-unsupported elements (silently skipped).
/// Returns `None` for harmless structural/LEMS tags we intentionally ignore
/// (e.g. `neuroml`, `Lems`, comments, `include` — handled elsewhere).
fn skipped_element_note(tag: &str) -> Option<&'static str> {
    Some(match tag {
        // STDP (в самом NeuroML помечен как «EXAMPLE NOT YET WORKING») — не
        // поддержан; blockingPlasticSynapse уже парсится выше.
        "stdpSynapse" => " — STDP-пластичность",
        // Dimensionless *DL-генераторы (ток задан в безразмерных единицах —
        // нельзя перевести в физический ток без клеточного масштаба).
        "pulseGeneratorDL" | "sineGeneratorDL" | "rampGeneratorDL" | "compoundInputDL" => {
            " — dimensionless (DL) input-генератор"
        }
        "doubleSynapse" => " — двойной синапс (два профиля на связи)",
        _ => return None,
    })
}

/// Parse a cell reference like `"pop[0]"` or `"../pop[0]"` into `("pop", 0)`.
pub(crate) fn parse_cell_ref(target: &str) -> Result<(&str, usize), NeuromlError> {
    let open = target
        .find('[')
        .ok_or_else(|| NeuromlError::parse(format!("bad cell reference `{target}`")))?;
    let close = target
        .find(']')
        .ok_or_else(|| NeuromlError::parse(format!("bad cell reference `{target}`")))?;
    // Guard against malformed brackets: `pop]0[` would panic on slice.
    if open >= close {
        return Err(NeuromlError::parse(format!(
            "bad cell reference `{target}` (brackets out of order)"
        )));
    }
    // Strip a leading path ("../") and keep the population id.
    let pop = target[..open].rsplit('/').next().unwrap_or(&target[..open]);
    let idx: usize = target[open + 1..close]
        .parse()
        .map_err(|_| NeuromlError::parse(format!("bad cell index in `{target}`")))?;
    Ok((pop, idx))
}
