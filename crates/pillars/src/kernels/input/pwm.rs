// ═══════════════════════════════════════════════════════════
//  PWM encoder
// ═══════════════════════════════════════════════════════════

const GOLDEN_RATIO: f32 = 0.618_034;

/// Pulse-Width Modulation encoder for sensor→spike conversion.
///
/// Each sensor gets a phase shift `(i·0.618034) mod 1` (golden-ratio hashing).
/// At step `t`, sensor `i` emits `spike_current` when `sensors[i] > (t/period +
/// shift) mod 1` — the duty cycle follows the sensor value.
pub struct PwmEncoder {
    num_sensors: usize,
    period: usize,
    tick: usize,
    spike_current: f32,
    phase_shifts: Vec<f32>,
}

impl PwmEncoder {
    pub fn new(n: usize, period: usize, spike_current: f32) -> Self {
        let phase_shifts: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * GOLDEN_RATIO).fract())
            .collect();
        Self {
            num_sensors: n,
            period,
            tick: 0,
            spike_current,
            phase_shifts,
        }
    }

    /// Encode sensor values into spike currents.
    pub fn encode(&mut self, sensors: &[f32], out_currents: &mut [f32]) {
        let t = self.tick as f32 / self.period as f32;
        self.tick = (self.tick + 1) % self.period;
        for i in 0..self.num_sensors.min(sensors.len()).min(out_currents.len()) {
            let threshold = (t + self.phase_shifts[i]).fract();
            out_currents[i] = if sensors[i] > threshold {
                self.spike_current
            } else {
                0.0
            };
        }
    }
}
