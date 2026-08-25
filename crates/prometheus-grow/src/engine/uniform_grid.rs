//! BioDynaMo `core/environment/uniform_grid_environment.h` port.
//! O(1) neighbor lookup via spatial hashing — replaces O(N²) in `form_synapses`.

use crate::Vec3;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct UniformGrid {
    pub cell_size: f64,
    pub inv_cell: f64,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl UniformGrid {
    pub fn new(cell_size: f64) -> Self {
        UniformGrid {
            cell_size,
            inv_cell: 1.0 / cell_size,
            buckets: HashMap::new(),
        }
    }
    fn key(&self, p: Vec3) -> (i32, i32, i32) {
        // Clamp to a huge-but-finite range: positions far outside the world
        // (e.g. a runaway growth cone) must not overflow the `cx+dx` sum in
        // `query` (latent panic in debug builds).
        let k = |x: f64| -> i32 {
            let v = (x * self.inv_cell).floor();
            if v.is_nan() {
                0
            } else {
                v.clamp(-1_000_000.0, 1_000_000.0) as i32
            }
        };
        (k(p.x), k(p.y), k(p.z))
    }
    pub fn clear(&mut self) {
        self.buckets.clear();
    }
    pub fn insert(&mut self, idx: usize, pos: Vec3) {
        self.buckets.entry(self.key(pos)).or_default().push(idx);
    }
    /// all indices within `radius` of `p`
    pub fn query(&self, p: Vec3, radius: f64) -> Vec<usize> {
        let r = (radius * self.inv_cell).ceil().min(1_000_000.0) as i32;
        let (cx, cy, cz) = self.key(p);
        let mut out = Vec::new();
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some(v) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        out.extend(v);
                    }
                }
            }
        }
        out
    }
}
