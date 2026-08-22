//! BioDynaMo `core/operation/operation.h` port
//! Operation = unit of work the Scheduler executes (diffuse, run modules, divide).

pub trait Operation {
    fn execute(&mut self, world: &mut crate::World);
}

pub struct Scheduler { ops: Vec<Box<dyn Operation>> }

impl Scheduler {
    pub fn new() -> Self { Scheduler { ops: Vec::new() } }
    pub fn add(&mut self, op: Box<dyn Operation>) { self.ops.push(op); }
    pub fn run(&mut self, world: &mut crate::World) {
        for op in self.ops.iter_mut() { op.execute(world); }
    }
}

// Example ops wrapping World methods
pub struct DiffuseOp;
impl Operation for DiffuseOp { fn execute(&mut self, w: &mut crate::World) { w.diffuse_substances(); } }
pub struct StepModulesOp;
impl Operation for StepModulesOp {
    fn execute(&mut self, w: &mut crate::World) {
        w.run_cell_modules(); w.process_divisions(); w.run_soma_modules(); w.run_neurite_modules();
        let d=w.synapse_dist; w.form_synapses(d);
    }
}
