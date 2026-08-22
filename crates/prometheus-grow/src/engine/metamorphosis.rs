//! Metamorphosis module: ecdysone pulse prunes weak synapses, then
//! adult-born neurons are added. Models larva -> pupa -> adult transition.
//!
//! Biology: ecdysteroid pulse (BROAD/Hr46) triggers:
//!   1. Prune weak/unused synapses (30-50% in Drosophila VNC)
//!   2. Adult-born neurons migrate to rind (dpp-dependent)
//!   3. Re-wire via Dscam1/CadN cell-recognition

use crate::cell::CellType;
use crate::module::Host;
use crate::Vec3;
use crate::module::Module;
use crate::world::World;

pub struct Metamorphosis {
    pub prune_fraction: f64,
    pub adult_born_per_step: usize,
    pub pupal_steps: usize,
    pulse_fired: bool,
    step: usize,
}

impl Metamorphosis {
    pub fn new(prune_fraction: f64, adult_born: usize, pupal_steps: usize) -> Self {
        Metamorphosis { prune_fraction, adult_born_per_step: adult_born, pupal_steps, pulse_fired: false, step: 0 }
    }
}

impl Module for Metamorphosis {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(_cell) = host else { return; };
        if self.step >= self.pupal_steps { return; }
        self.step += 1;

        if !self.pulse_fired && self.step == 1 {
            self.pulse_fired = true;
            if self.prune_fraction > 0.0 && !world.synapses.is_empty() {
                let n_prune = (world.synapses.len() as f64 * self.prune_fraction) as usize;
                let mut ws: Vec<(usize, f64)> = world.synapses.iter().enumerate().map(|(i, s)| (i, s.weight)).collect();
                ws.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut remove: Vec<usize> = ws.iter().take(n_prune).map(|(i, _)| *i).collect();
                remove.sort_by(|a, b| b.cmp(a));
                for idx in remove { world.synapses.swap_remove(idx); }
            }
        }

        if self.pulse_fired && self.adult_born_per_step > 0 {
            let center = Vec3::new(
                (world.bounds.min.x + world.bounds.max.x) * 0.5,
                (world.bounds.min.y + world.bounds.max.y) * 0.5,
                (world.bounds.min.z + world.bounds.max.z) * 0.5,
            );
            let span = world.bounds.max - world.bounds.min;
            for _ in 0..self.adult_born_per_step {
                let d = Vec3::new(world.rand()-0.5, world.rand()-0.5, world.rand()-0.5).normalized();
                let r = 0.35 + world.rand() * 0.15;
                let max_dim = span.x.max(span.y).max(span.z);
                let pos = center + d * r * max_dim;
                let pos = Vec3::new(
                    pos.x.clamp(world.bounds.min.x, world.bounds.max.x),
                    pos.y.clamp(world.bounds.min.y, world.bounds.max.y),
                    pos.z.clamp(world.bounds.min.z, world.bounds.max.z),
                );
                world.add_cell(pos, CellType::Excitatory, 3.0);
            }
        }
    }

    fn copy_on_divide(&self) -> bool { false }
    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(Metamorphosis { prune_fraction: self.prune_fraction, adult_born_per_step: self.adult_born_per_step, pupal_steps: self.pupal_steps, pulse_fired: self.pulse_fired, step: self.step })
    }
}
