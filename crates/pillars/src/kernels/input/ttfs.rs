// ═══════════════════════════════════════════════════════════
//  TTFS (Time-To-First-Spike)
// ═══════════════════════════════════════════════════════════

/// TTFS: encodes a value in [0, 1] as a spike delayed by `(1 − value)·max_delay`
/// steps — the larger the value, the earlier the spike.
pub struct TtfsEncoder {
    pub max_delay: usize,
    pub current_step: usize,
    pub fire_times: Vec<usize>,
    pub fired: Vec<bool>,
}

impl TtfsEncoder {
    pub fn new(n: usize, max_delay: usize) -> Self {
        Self {
            max_delay,
            current_step: 0,
            fire_times: vec![0; n],
            fired: vec![false; n],
        }
    }

    /// Encode values into TTFS spikes over multiple steps.
    /// Returns `true` if there are active spikes on this step.
    pub fn encode_step(&mut self, values: &[f32], spikes_out: &mut [i32]) -> bool {
        let n = spikes_out
            .len()
            .min(values.len())
            .min(self.fire_times.len());
        let mut any_spike = false;

        if self.current_step == 0 {
            for i in 0..n {
                let val = values[i].clamp(0.0, 1.0);
                self.fire_times[i] = ((1.0 - val) * self.max_delay as f32) as usize;
                self.fired[i] = false;
            }
        }

        for i in 0..n {
            if !self.fired[i] && self.current_step >= self.fire_times[i] {
                spikes_out[i] = 1;
                self.fired[i] = true;
                any_spike = true;
            } else {
                spikes_out[i] = 0;
            }
        }

        self.current_step += 1;
        if self.current_step > self.max_delay {
            self.current_step = 0;
        }
        any_spike
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
        self.fired.fill(false);
        self.fire_times.fill(0);
    }
}
