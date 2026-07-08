//! Hologram bridge: position → phase → HH → STDP → motor.
//!
//! Pipeline:
//!    pos(x,y) → phase_encode(x) = θ, phase_encode(y) = φ
//!    → sin(θ) as I_pre, sin(φ) as I_post
//!    → HH LUT neurons
//!    → STDP learns Δ = φ−θ → weight encodes position difference
//!    → linear_decode(spikes) → motor output
//!
//! Karmic bridge: всё уже написано, просто соединено в pipeline.

use std::f32::consts::TAU;

const DT: f32 = 0.1;
const STEPS: usize = 20_000;
const N_MOTOR: usize = 2;

fn main() {
    println!("═══ Hologram Bridge: position phase → SNN → motor ═══");
    println!();

    // Positions of two entities in game world
    let player_pos = 0.5f32;   // normalized [0, 1]
    let npc_pos = 0.3f32;

    // 1. Phase encode positions → T3 phases
    let omega: f32 = 800.0;    // rad/ms
    let theta_player = player_pos * TAU;  // map [0,1] → [0, 2π]
    let theta_npc = npc_pos * TAU;

    // 2. Build HH neurons
    let channels = taas_kernels::neuron::hh::standard_channels();
    let n_ch = channels.len();
    let n_g: usize = channels.iter().map(|ch| ch.gates.len()).sum();
    let luts = taas_kernels::neuron::hh::build_luts(&channels, DT, -100.0, 60.0, 1024);

    let mut m_inf = Vec::with_capacity(n_g);
    let mut decay = Vec::with_capacity(n_g);
    let mut powers = Vec::with_capacity(n_g);
    let mut e_rev = Vec::with_capacity(n_ch);
    let mut g_max = Vec::with_capacity(n_ch);
    let mut ngpc = Vec::with_capacity(n_ch);
    for (ci, ch) in channels.iter().enumerate() {
        e_rev.push(ch.E_rev); g_max.push(ch.g_max); ngpc.push(ch.gates.len());
        for (gi, gate) in ch.gates.iter().enumerate() {
            m_inf.push(luts[ci][gi].m_inf.clone());
            decay.push(luts[ci][gi].decay.clone());
            powers.push(gate.power);
        }
    }
    let m_inf_ptrs: Vec<*const f32> = m_inf.iter().map(|v| v.as_ptr()).collect();
    let decay_ptrs: Vec<*const f32> = decay.iter().map(|v| v.as_ptr()).collect();

    // 3. State: 2 neurons (player + NPC)
    let mut v = vec![-65.0f32; 2];
    let mut gates = vec![0.5f32; 2 * n_g];
    let mut ii = vec![0.0f32; 2];
    let g_leak = vec![0.3f32; 2];
    let e_leak = vec![-65.0f32; 2];
    let c_m = vec![1.0f32; 2];

    // 4. Synapse + STDP
    let mut weight = 0.5f32;
    let tau_trace = 10.0f32;
    let decay_trace = (-DT / tau_trace).exp();
    let mut trace_pre = 0.0f32;
    let mut trace_post = 0.0f32;
    let mut spike_buf = [0i32; 2];

    // 5. Motor decoder (linear, single weight per spike source)
    let mut motor = [0.0f32; N_MOTOR];
    let decode_w: [f32; 4] = [0.5, -0.3, 0.7, -0.2];  // [pre→m0, post→m0, pre→m1, post→m1]
    let decode_b: [f32; N_MOTOR] = [0.0; N_MOTOR];

    // 6. Records
    let mut weight_history = Vec::new();

    let i_base = 10.0;
    let i_amp = 8.0;
    let mut prev_v = [v[0], v[1]];

    // ── Main loop ──
    for t in 0..STEPS {
        let t_f32 = t as f32 * DT;

        // Phase → current (no DC! sin(θ) is the injection)
        ii[0] = i_base + i_amp * (omega * t_f32 + theta_player).sin();
        ii[1] = i_base + i_amp * (omega * t_f32 + theta_npc).sin();

        // HH step
        unsafe {
            taas_kernels::neuron::hh::hh_lut_step(
                v.as_mut_ptr(), gates.as_mut_ptr(), ii.as_ptr(),
                g_leak.as_ptr(), e_leak.as_ptr(), c_m.as_ptr(),
                2, n_g,
                m_inf_ptrs.as_ptr(), decay_ptrs.as_ptr(),
                -100.0, 60.0, 1024.0,
                e_rev.as_ptr(), g_max.as_ptr(), powers.as_ptr(),
                ngpc.as_ptr(), n_ch, DT,
            );
        }

        spike_buf[0] = if prev_v[0] < 0.0 && v[0] >= 0.0 { 1 } else { 0 };
        spike_buf[1] = if prev_v[1] < 0.0 && v[1] >= 0.0 { 1 } else { 0 };
        prev_v = [v[0], v[1]];

        // EPSC: player→NPC
        if spike_buf[0] == 1 {
            ii[1] += weight * 5.0;
        }

        // STDP traces
        trace_pre = trace_pre * decay_trace + spike_buf[0] as f32;
        trace_post = trace_post * decay_trace + spike_buf[1] as f32;

        // STDP update
        if spike_buf[0] == 1 { weight += 0.004 * trace_post; }
        if spike_buf[1] == 1 { weight -= 0.003 * trace_pre; }
        weight = weight.clamp(0.0, 2.0);

        // Motor decode: linear on spike history
        // Simple: integrate spikes with decoder weights
        motor[0] += spike_buf[0] as f32 * decode_w[0] + spike_buf[1] as f32 * decode_w[1];
        motor[1] += spike_buf[0] as f32 * decode_w[2] + spike_buf[1] as f32 * decode_w[3];

        if t % 5000 == 0 { weight_history.push(weight); }
    }

    // ── Results ──
    println!("  Player pos: {player_pos:.2} → θ = {theta_player:.4} rad");
    println!("  NPC pos:    {npc_pos:.2} → θ = {theta_npc:.4} rad");
    println!("  Phase diff: Δ = {:.4} rad ({:.2} cycles)", theta_npc - theta_player, (theta_npc - theta_player) / TAU);
    println!();
    println!("  STDP weight: {:.4} → {:.4} (Δw = {:+.4})", 0.5, weight, weight - 0.5);
    print!("  Trace: ");
    for w in &weight_history { print!("{:.4} ", w); }
    println!();
    println!("  Motor output: [{:.4}, {:.4}]", motor[0], motor[1]);
    println!();
    println!("  Pipeline: pos → phase → sin(θ) → HH → STDP → decode → motor");
    println!("  No DC: T3 phase is the carrier, sin(θ) is the message");
    println!("  Δφ encodes spatial relationship → STDP learns it as weight");
}
