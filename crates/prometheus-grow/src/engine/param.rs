//! BioDynaMo `core/param/param.h` port — centralized, CLI-validated parameters.

#[derive(Clone, Debug)]
pub struct Param {
    pub dt: f64,
    pub max_cells: usize,
    pub max_neurites: usize,
    pub synapse_dist: f64,
    pub synapse_strength: f64,
    pub diffusion_dc: f64,
    pub degradation_mu: f64,
}

impl Default for Param {
    fn default() -> Self {
        Param {
            dt: 1.0,
            max_cells: 64,
            max_neurites: 512,
            synapse_dist: 80.0,
            synapse_strength: 7.0,
            diffusion_dc: 0.1,
            degradation_mu: 0.001,
        }
    }
}
impl Param {
    pub fn validate(&self) -> Result<(), String> {
        if self.dt <= 0.0 {
            return Err("dt>0".into());
        }
        if self.synapse_dist <= 0.0 {
            return Err("synapse_dist>0".into());
        }
        Ok(())
    }
}
