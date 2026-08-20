use super::*;

/// Passive cable solver over a [`CompartmentTree`] — implicit Euler (Hines).
///
/// Solves `C_i·dV_i/dt = -g_leak,i·(V_i − E) + Σ_j (V_j − V_i)/R_ij + I_i` with
/// backward Euler for the linear passive system, so it is unconditionally
/// stable. The tree-structured linear system is solved in O(n) by the Hines
/// algorithm (upward elimination + downward back-substitution).
#[derive(Clone)]
pub(crate) struct CableSolver {
    tree: CompartmentTree,
    /// Membrane potential per compartment (mV).
    v: Vec<f32>,
    e_leak: f32,
    /// Fixed step (ms). Only read by the test-only [`step`](Self::step);
    /// production stepping goes through [`step_with_extra`](Self::step_with_extra).
    #[allow(dead_code)]
    dt: f32,
    /// Children adjacency: `children[i]` = compartments whose parent is `i`.
    children: Vec<Vec<usize>>,
    /// Topological order (root first) for the back-substitution pass.
    order: Vec<usize>,
    /// Axial conductance to parent (µS); 0 for the root. **Geometry-only** —
    /// built once at construction, reused every step (no divisions per step).
    g_parent: Vec<f32>,
    /// Σ g_parent[child] over direct children (µS). Geometry-only.
    g_children_sum: Vec<f32>,
    /// Reused implicit-solve buffers (sized `n` — no per-step allocations).
    diag: Vec<f32>,
    rhs: Vec<f32>,
    /// Scratch for the back-substitution; swapped with `v` each step (zero
    /// copy).
    v_new: Vec<f32>,
}

impl CableSolver {
    /// Build a solver at rest (`V = e_leak` everywhere).
    pub fn new(tree: CompartmentTree, e_leak: f32, dt: f32) -> Self {
        let n = tree.len();
        let mut children = vec![Vec::new(); n];
        for (i, c) in tree.compartments.iter().enumerate() {
            if let Some(p) = c.parent {
                children[p].push(i);
            }
        }
        // BFS topological order from the root.
        let mut order = Vec::with_capacity(n);
        let mut queue = vec![tree.root];
        let mut head = 0usize;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            order.push(i);
            queue.extend_from_slice(&children[i]);
        }
        // Cache the axial conductances (they never change with the state).
        let mut g_parent = vec![0.0f32; n];
        for (i, c) in tree.compartments.iter().enumerate() {
            if c.parent.is_some() {
                g_parent[i] = 1.0 / tree.junction_resistance(i);
            }
        }
        let mut g_children_sum = vec![0.0f32; n];
        for (i, childs) in children.iter().enumerate() {
            g_children_sum[i] = childs.iter().map(|&c| g_parent[c]).sum();
        }
        Self {
            tree,
            v: vec![e_leak; n],
            e_leak,
            dt,
            children,
            order,
            g_parent,
            g_children_sum,
            diag: vec![0.0; n],
            rhs: vec![0.0; n],
            v_new: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Index of the root compartment (the soma). Reached from production via
    /// [`ActiveCableSolver::root`]; kept un-gated so the method survives when
    /// `ActiveCableSolver` is compiled without its own gating.
    #[allow(dead_code)]
    pub fn root(&self) -> usize {
        self.tree.root
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.v[idx]
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.v[idx] = value;
    }

    /// One implicit-Euler (Hines) step of the passive cable. Test-only —
    /// production stepping goes through [`step_with_extra`](Self::step_with_extra)
    /// (which carries the extra conductance and an explicit `dt`).
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    #[cfg(test)]
    pub fn step(&mut self, i_inj: &[f32]) {
        self.step_with_extra(i_inj, &[], &[], self.dt);
    }

    /// Like [`step`](Self::step) but adds per-compartment extra conductances
    /// `g_extra` (µS) and `g_extra·E_rev` (`g_e_extra`, nA) — the semi-implicit
    /// active-channel contribution (gates evaluated at the old voltage). Empty
    /// slices mean no extra conductance.
    ///
    /// # Panics
    /// If `i_inj` (or, when non-empty, `g_extra`/`g_e_extra`) length mismatch.
    pub fn step_with_extra(&mut self, i_inj: &[f32], g_extra: &[f32], g_e_extra: &[f32], dt: f32) {
        assert_eq!(i_inj.len(), self.tree.len(), "i_inj length mismatch");
        assert!(g_extra.is_empty() || g_extra.len() == self.tree.len());
        assert!(g_e_extra.is_empty() || g_e_extra.len() == self.tree.len());
        let n = self.tree.len();
        if n == 0 {
            return;
        }
        let e_leak = self.e_leak;
        let has_ge = !g_extra.is_empty();
        let has_gee = !g_e_extra.is_empty();

        // Diagonal (µS) and RHS (nA) of the implicit system A·V = b. All
        // buffers are reused across steps (zero per-step allocations); the
        // geometry-only conductances come from the cached `g_parent` /
        // `g_children_sum`.
        for i in 0..n {
            let c = &self.tree.compartments[i];
            let inv_dt = c.cm_nf / dt; // nF/ms = µS
            let ge = if has_ge { g_extra[i] } else { 0.0 };
            let gee = if has_gee { g_e_extra[i] } else { 0.0 };
            self.diag[i] = inv_dt + c.g_leak_us + self.g_parent[i] + self.g_children_sum[i] + ge;
            self.rhs[i] = inv_dt * self.v[i] + c.g_leak_us * e_leak + i_inj[i] + gee;
        }

        // Hines elimination: children before parents (reverse BFS order).
        for &i in self.order.iter().rev() {
            for &child in &self.children[i] {
                let factor = self.g_parent[child] / self.diag[child];
                self.diag[i] -= self.g_parent[child] * factor;
                self.rhs[i] += factor * self.rhs[child];
            }
        }

        // Back-substitution: parents before children (BFS order), into the
        // scratch `v_new`, then zero-copy swap with `v`.
        for &i in &self.order {
            let up = match self.tree.compartments[i].parent {
                Some(p) => self.g_parent[i] * self.v_new[p],
                None => 0.0,
            };
            self.v_new[i] = (self.rhs[i] + up) / self.diag[i];
        }
        std::mem::swap(&mut self.v, &mut self.v_new);
    }
}
