//! Симуляция — шаг, сброс, история Scope.

use std::collections::VecDeque;

use pillars::api::formula::{self, I_SYN, N_REGS, S0, S1, S2, S3};

use super::NeuronBuilder;

impl NeuronBuilder {
    pub(crate) fn build_program(&self) -> Vec<pillars::api::formula::OpNode> {
        super::compile::graph_to_program(&self.nodes, &self.edges)
    }
    pub(crate) fn input_reg(&self, node: usize, pin: usize) -> usize {
        super::compile::input_reg_of(&self.nodes, &self.edges, node, pin)
    }
    pub(crate) fn scope_reg(&self) -> usize {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.kind == pillars::api::formula::OpKind::Scope {
                return self.input_reg(i, 0);
            }
        }
        S0
    }
    pub(crate) fn trace_reg(&self) -> usize {
        match self.trace {
            1 => pillars::api::formula::S0,
            2 => pillars::api::formula::S1,
            3 => pillars::api::formula::S2,
            4 => pillars::api::formula::S3,
            _ => self.scope_reg(),
        }
    }
    pub(crate) fn step_sim(&mut self) {
        let prog = self.build_program();
        formula::run_program::<f32>(&prog, &mut self.regs, self.dt, self.i_syn);
        self.t += self.dt as f64;
        let reg = self.trace_reg();
        let cur = self.regs[reg];
        if self.prev_trace < self.spike_thresh && cur >= self.spike_thresh {
            self.total_spikes += 1;
            self.spike_times.push(self.t);
            if self.spike_times.len() > 128 {
                self.spike_times.drain(..self.spike_times.len() - 128);
            }
        }
        self.prev_trace = cur;
        self.history.push_back((self.t, cur as f64));
        while self.history.len() > super::geometry::SCOPE_POINTS {
            self.history.pop_front();
        }
    }
    pub(crate) fn reset_sim(&mut self) {
        self.regs = [0.0; N_REGS];
        self.regs[S0] = self.v_init;
        self.regs[pillars::api::formula::S1] = self.init_gates[0];
        self.regs[pillars::api::formula::S2] = self.init_gates[1];
        self.regs[pillars::api::formula::S3] = self.init_gates[2];
        self.t = 0.0;
        self.history.clear();
        self.spike_times.clear();
        self.total_spikes = 0;
        self.prev_trace = self.v_init;
    }
}
