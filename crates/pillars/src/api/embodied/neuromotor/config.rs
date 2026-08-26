use super::*;

/// Physiological profile for muscle tissue scaling and biomechanics.
#[derive(Debug, Clone, Copy)]
pub struct BiomechanicsProfile {
    pub d_ca: f32,
    pub tau_pump: f32,
    pub eta_reset: f32,
    pub t_max: f32,
    pub kd: f32,
    pub l0: f32,
    pub v_max: f32,
    pub k0: f32,
    pub beta: f32,
    pub b0: f32,
    pub alpha: f32,
}

impl BiomechanicsProfile {
    /// Mammalian-scale profile (macroscopic forces in Newtons).
    pub const fn mammal() -> Self {
        Self {
            d_ca: 0.5,
            tau_pump: 50.0,
            eta_reset: 0.8,
            t_max: 10.0, // 10 N
            kd: 0.3,
            l0: 1.0,
            v_max: 5.0,
            k0: 1.0,
            beta: 2.0,
            b0: 0.1,
            alpha: 0.5,
        }
    }

    /// Insect-scale profile (microscopic forces in milli-Newtons).
    pub const fn insect() -> Self {
        Self {
            d_ca: 0.5,
            tau_pump: 20.0,
            eta_reset: 0.8,
            t_max: 0.001, // 1 mN
            kd: 0.3,
            l0: 1.0,
            v_max: 0.5,
            k0: 1e-5,
            beta: 1e-4,
            b0: 1e-6,
            alpha: 1e-5,
        }
    }

    /// Drosophila-scale profile (forces in micro-Newtons).
    pub const fn drosophila() -> Self {
        Self {
            d_ca: 0.3,
            tau_pump: 10.0,
            eta_reset: 0.9,
            t_max: 2e-6, // 2 µN — intermediate fiber
            kd: 0.2,
            l0: 0.0003, // 300 µm fiber length
            v_max: 0.5,
            k0: 1e-8,
            beta: 1e-7,
            b0: 1e-9,
            alpha: 1e-8,
        }
    }

    /// Drosophila slow fiber pool — postural tone (~0.1 µN per spike).
    pub const fn drosophila_slow() -> Self {
        Self {
            d_ca: 0.1,
            tau_pump: 20.0,
            eta_reset: 0.9,
            t_max: 1e-7, // 0.1 µN
            kd: 0.15,
            l0: 0.0002, // 200 µm
            v_max: 0.1,
            k0: 1e-8,
            beta: 1e-8,
            b0: 1e-9,
            alpha: 1e-9,
        }
    }

    /// Drosophila fast fiber pool — ballistic (~10 µN per spike).
    pub const fn drosophila_fast() -> Self {
        Self {
            d_ca: 1.0,
            tau_pump: 8.0,
            eta_reset: 0.7,
            t_max: 1e-5, // 10 µN
            kd: 0.3,
            l0: 0.0004, // 400 µm
            v_max: 1.0,
            k0: 1e-8,
            beta: 1e-7,
            b0: 1e-9,
            alpha: 1e-8,
        }
    }

    /// TDT (Tergal Depressor of Trochanter) — catapult muscle for jumping.
    pub const fn drosophila_tdt() -> Self {
        Self {
            d_ca: 2.0,      // massive Ca²⁺ injection
            tau_pump: 50.0, // slow pump
            eta_reset: 0.5, // passive reset
            t_max: 2e-4,    // 200 µN — 100× body weight!
            kd: 0.1,
            l0: 0.0005, // 500 µm — spans thorax
            v_max: 2.0, // 2 m/s — ballistic
            k0: 1e-7,
            beta: 1e-6,
            b0: 1e-8,
            alpha: 1e-7,
        }
    }
}

impl Default for BiomechanicsProfile {
    fn default() -> Self {
        Self::mammal()
    }
}

/// Configuration for a single Drosophila muscle fiber pool.
#[derive(Debug, Clone, Copy)]
pub struct FiberPoolConfig {
    /// Number of fibers in this pool (typically 3-7 per fly muscle).
    pub n_fibers: usize,
    /// Biophysical profile for this pool.
    pub profile: BiomechanicsProfile,
}

impl FiberPoolConfig {
    /// Slow postural fibers: 5-7 fibers, ~0.1 µN each.
    pub const fn slow() -> Self {
        Self {
            n_fibers: 6,
            profile: BiomechanicsProfile::drosophila_slow(),
        }
    }
    /// Intermediate walking fibers: 4-6 fibers, ~2 µN each.
    pub const fn intermediate() -> Self {
        Self {
            n_fibers: 5,
            profile: BiomechanicsProfile::drosophila(),
        }
    }
    /// Fast ballistic fibers: 3-5 fibers, ~10 µN each.
    pub const fn fast() -> Self {
        Self {
            n_fibers: 4,
            profile: BiomechanicsProfile::drosophila_fast(),
        }
    }
    /// TDT catapult: 3 super-thick fibers, ~200 µN each.
    pub const fn tdt() -> Self {
        Self {
            n_fibers: 3,
            profile: BiomechanicsProfile::drosophila_tdt(),
        }
    }
}

/// Full Drosophila leg musculature configuration.
#[derive(Debug, Clone)]
pub struct DrosophilaLegConfig {
    /// Number of legs.
    pub n_legs: usize,
    /// Number of muscles per leg.
    pub muscles_per_leg: usize,
    /// Number of joints per leg.
    pub joints_per_leg: usize,
    /// Total muscles across all legs.
    pub total_muscles: usize,
    /// Total fiber channels (sum of all pool.n_fibers).
    pub total_channels: usize,
    /// Whether to include TDT catapult muscles (middle legs only).
    pub include_tdt: bool,
    /// Fiber pool assignments for each muscle [n_muscles × pools].
    pub pools: Vec<Vec<FiberPoolConfig>>,
}

impl Default for DrosophilaLegConfig {
    fn default() -> Self {
        Self::hexapod()
    }
}

impl DrosophilaLegConfig {
    /// Standard 6-legged fly configuration.
    ///
    /// Each leg: 19 muscles × [slow, intermediate, fast] = 57 channels.
    /// Middle legs (T2, 2 legs): +3 TDT fibers each = +6 channels.
    /// Total: 6 × 57 + 6 = 348 channels → padded to 352 (44×8).
    pub fn hexapod() -> Self {
        let n_legs = 6;
        let muscles_per_leg = 19;
        let joints_per_leg = 5;

        let standard_pools = vec![
            FiberPoolConfig::slow(),
            FiberPoolConfig::intermediate(),
            FiberPoolConfig::fast(),
        ];

        let mut pools = Vec::new();
        for leg in 0..n_legs {
            for muscle in 0..muscles_per_leg {
                let mut muscle_pools = standard_pools.clone();
                // TDT is a SINGLE muscle (muscle 18) in middle legs only
                if (leg == 1 || leg == 2) && muscle == muscles_per_leg - 1 {
                    muscle_pools.push(FiberPoolConfig::tdt());
                }
                pools.push(muscle_pools);
            }
        }

        let total_channels: usize = pools
            .iter()
            .flat_map(|p| p.iter())
            .map(|pool| pool.n_fibers)
            .sum();

        // Pad to multiple of 8 for SIMD
        let n_padded = total_channels.div_ceil(8) * 8;

        Self {
            n_legs,
            muscles_per_leg,
            joints_per_leg,
            total_muscles: muscles_per_leg * n_legs,
            total_channels: n_padded, // store padded count for SIMD alignment
            include_tdt: true,
            pools,
        }
    }

    /// Build an NeuromotorState for the full fly with per-fiber parameters.
    pub fn build_state(&self) -> NeuromotorState {
        let n_channels = self.total_channels; // already padded to ×8

        let mut state = NeuromotorStateBuilder::new(n_channels)
            .with_profile(BiomechanicsProfile::drosophila())
            .build();

        let mut idx = 0;
        for muscle_pools in &self.pools {
            for pool in muscle_pools {
                for _ in 0..pool.n_fibers {
                    if idx < n_channels {
                        state.d_ca[idx] = pool.profile.d_ca;
                        state.tau_pump[idx] = pool.profile.tau_pump;
                        state.eta_reset[idx] = pool.profile.eta_reset;
                        state.t_max[idx] = pool.profile.t_max;
                        state.kd[idx] = pool.profile.kd;
                        state.l0[idx] = pool.profile.l0;
                        state.v_max[idx] = pool.profile.v_max;
                    }
                    idx += 1;
                }
            }
        }

        state
    }
}
