use super::basal_ganglia::{BasalGanglia, BgConfig};
use super::gnw::GlobalWorkspace;
use super::working_memory::WorkingMemory;

/// Configuration for MinBrain.
#[derive(Debug, Clone)]
pub struct MinBrainConfig {
    pub sensory_dim: usize,
    pub workspace_size: usize,
    pub pfc_slots: usize,
    pub n_actions: usize,
    /// Threshold to clear the working memory (0.0 to 1.0 threshold on clear action drive).
    pub clear_threshold: f32,
}

impl Default for MinBrainConfig {
    fn default() -> Self {
        Self {
            sensory_dim: 128,
            workspace_size: 64,
            pfc_slots: 4,
            n_actions: 8,
            clear_threshold: 0.8,
        }
    }
}

/// Minimal Brain Model (MinBrain).
///
/// Composed of:
/// - Prefrontal Cortex (Working Memory)
/// - Global Neuronal Workspace (GNW)
/// - Basal Ganglia (Action Selection)
pub struct MinBrain {
    pub config: MinBrainConfig,
    pub pfc: WorkingMemory,
    pub gnw: GlobalWorkspace,
    pub basal_ganglia: BasalGanglia,

    // Internal buffers
    gnw_input_buf: Vec<f32>,
}

impl MinBrain {
    /// Create a new MinBrain with random initialized connections.
    pub fn new(config: MinBrainConfig) -> Self {
        let pfc = WorkingMemory::new(config.pfc_slots, config.sensory_dim);

        let gnw_input_dim = config.sensory_dim * 2;
        let gnw = GlobalWorkspace::new(gnw_input_dim, config.workspace_size);

        let bg_config = BgConfig {
            n_cortical: gnw_input_dim,
            n_actions: config.n_actions,
            ..Default::default()
        };
        let basal_ganglia = BasalGanglia::with_config(bg_config);

        Self {
            config: config.clone(),
            pfc,
            gnw,
            basal_ganglia,
            gnw_input_buf: vec![0.0; gnw_input_dim],
        }
    }

    /// Run one tick of the MinBrain.
    ///
    /// Returns the selected action ID.
    pub fn step(&mut self, sensory_input: &[f32]) -> usize {
        assert_eq!(sensory_input.len(), self.config.sensory_dim);

        // 1. Tick PFC to allow recurrent decay & attractor dynamics.
        self.pfc.tick();

        // 2. Retrieve current context from PFC (slot 0 for simplicity)
        let pfc_context = self.pfc.recall(0);

        // 3. Assemble GNW input: [Sensory, PFC]
        for i in 0..self.config.sensory_dim {
            self.gnw_input_buf[i] = sensory_input[i];
            self.gnw_input_buf[self.config.sensory_dim + i] = pfc_context[i];
        }

        // 4. Run GNW Forward pass (updates gnw_input_buf in-place)
        self.gnw.forward(&mut self.gnw_input_buf);

        // 5. Basal Ganglia Action Selection based on global state
        let action = self.basal_ganglia.step(&self.gnw_input_buf);

        // 6. Action side-effects
        // Convention: action 0 = "Clear PFC"
        // Convention: action 1 = "Store sensory to PFC"
        if action == 0 {
            self.pfc.clear();
        } else if action == 1 {
            self.pfc.store(sensory_input);
        }

        action
    }

    /// Get the current PFC state.
    pub fn pfc_state(&self) -> Vec<f32> {
        self.pfc.recall(0)
    }
}
