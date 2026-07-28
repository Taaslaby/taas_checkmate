# Phase 1: FlyWire Embodied Test Architecture

## Data Flow

```
FlyWire Connectome (139K neurons, 80M synapses)
    ↓
neuron-engine-rs: LIF/Izhikevich spike generation
    ↓
Motor Neuron Outputs: spikes[motoneuron_idx] ∈ {0, 1}
    ↓
DelayBuffer: NMJ synaptic delay (1.5 ms)
    ↓
atlasov_neuromotor_step (SIMD f32x8):
    - Calcium injection: Ca += spikes[i] * ΔCa
    - Decay & active reset: Ca *= decay * (1 - reset_signal)
    - Hill saturation: T = T_max * Ca²/(Ca² + K_d²)
    ↓
Output: [ΔF, K_joint, B_joint] for each joint j
    ↓
MuJoCo Physics Loop (dt=1ms):
    1. Read: q_j, q̇_j from mjData
    2. Compute: τ_applied = ΔF - K·(q - q_rest) - B·q̇
    3. Write: data.qfrc_applied[dof_idx] = τ_applied
    4. Step: mj_step() advances physics by 1ms
    ↓
Kinematics → Video/Analysis
```

## Module Structure (Rust)

```
taas_embodied/  [NEW]
├── lib.rs
│   └── pub mod fly_embodied;
├── fly_embodied.rs
│   ├── struct FlyEmbodiedState {
│   │   pub neuron_engine: NeuronEngine,
│   │   pub atlasov_state: AtlasovState,
│   │   pub delay_buffers: Vec<DelayBuffer>,
│   │   pub mujoco_model: *mut mjModel,
│   │   pub mujoco_data: *mut mjData,
│   │   pub motoneuron_indices: Vec<usize>,
│   │   pub joint_mappings: Vec<JointMapping>,
│   │   pub step_count: u64,
│   │ }
│   │
│   ├── impl FlyEmbodiedState {
│   │   pub fn new(xml_path: &str) -> Self { }
│   │   pub fn step(&mut self, dt: f32) -> SimStepOutput { }
│   │   pub fn get_pose(&self) -> FlyPose { }
│   │   pub fn get_motoneuron_firing_rates(&self) -> Vec<f32> { }
│   │ }
│   │
│   └── struct JointMapping {
│       pub dof_idx: usize,           // MuJoCo DOF
│       pub agonist_muscle: usize,    // AtlasovState index
│       pub antagonist_muscle: usize,
│       pub q_rest: f32,
│   }
│
├── tests/
│   └── embodied_tests.rs
│       ├── test_spontaneous_walking
│       ├── test_posture_stability
│       ├── test_leg_coordination
│       └── test_neural_motor_latency
│
└── benches/
    └── embodied_benchmark.rs
```

## Key Test Cases

### Test 1: Spontaneous Walking Pattern
```rust
#[test]
fn test_spontaneous_walking() {
    let mut fly = FlyEmbodiedState::new("fruitfly.xml");
    
    // Record leg angles for 10 seconds (10,000 steps at 1ms)
    let mut leg_angles = Vec::new();
    for _ in 0..10_000 {
        fly.step(0.001);
        let pose = fly.get_pose();
        leg_angles.push(pose.front_left_hip);
        leg_angles.push(pose.mid_left_hip);
        leg_angles.push(pose.back_left_hip);
        // + right legs
    }
    
    // Analysis: do legs show alternating pattern?
    // Expected: tripod gait (front+back vs middle alternates)
    let alternation_score = compute_alternation(&leg_angles);
    assert!(alternation_score > 0.7, "Legs should alternate");
}
```

### Test 2: Stability During Perturbation
```rust
#[test]
fn test_perturbation_recovery() {
    let mut fly = FlyEmbodiedState::new("fruitfly.xml");
    
    // Walk for 2 seconds
    for _ in 0..2000 { fly.step(0.001); }
    
    // Apply sudden external force
    fly.mujoco_data.qfrc_applied[0] += 0.05; // 50mN impulse
    
    // Check: does fly return to stable walking?
    let mut fall_count = 0;
    for step in 0..5000 {
        fly.step(0.001);
        let pose = fly.get_pose();
        if pose.body_height < 0.8 { fall_count += 1; }
    }
    
    assert!(fall_count < 500, "Fly should recover from perturbation");
}
```

### Test 3: Neural Latency Verification
```rust
#[test]
fn test_motor_latency_end_to_end() {
    let mut fly = FlyEmbodiedState::new("fruitfly.xml");
    
    // Inject current into specific descending command neuron
    let dcn_idx = fly.motoneuron_indices[0];
    fly.neuron_engine.inject_current(dcn_idx, 5.0); // 5 nA
    
    // Measure: when does leg start moving?
    let mut spike_time = None;
    let mut movement_time = None;
    
    for step in 0..500 { // 500ms window
        fly.step(0.001);
        
        if spike_time.is_none() && fly.neuron_engine.did_spike(dcn_idx) {
            spike_time = Some(step as f32 * 0.001);
        }
        
        let pose = fly.get_pose();
        if movement_time.is_none() && pose.front_right_hip.abs() > 0.05 {
            movement_time = Some(step as f32 * 0.001);
        }
    }
    
    let latency = movement_time.unwrap() - spike_time.unwrap();
    assert!((latency - 0.002).abs() < 0.0005, // Should be ~2ms ± 0.5ms
            "Motor latency should be ~2ms (NMJ delay)");
}
```

## MuJoCo Integration Pattern

```rust
impl FlyEmbodiedState {
    pub fn step(&mut self, dt: f32) -> SimStepOutput {
        // === Step 1: Neural simulation (10µs) ===
        self.neuron_engine.step(dt);
        let motoneuron_spikes = self.neuron_engine.get_spikes(&self.motoneuron_indices);
        
        // === Step 2: NMJ synaptic delay (1.5ms) ===
        let delayed_spikes = self.delay_buffers.iter_mut().zip(motoneuron_spikes.iter())
            .map(|(buf, spike)| buf.push(*spike))
            .collect::<Vec<_>>();
        
        // === Step 3: Muscle dynamics (100µs) ===
        self.atlasov_state.step(&delayed_spikes, dt);
        let (forces, stiffness, damping) = self.atlasov_state.get_outputs();
        
        // === Step 4: Apply forces to MuJoCo (50µs) ===
        for (joint_map, &df, &k, &b) in izip!(
            &self.joint_mappings,
            forces.iter(),
            stiffness.iter(),
            damping.iter(),
        ) {
            let q = self.mujoco_data.qpos[joint_map.dof_idx];
            let q_dot = self.mujoco_data.qvel[joint_map.dof_idx];
            
            let tau_applied = df - k * (q - joint_map.q_rest) - b * q_dot;
            self.mujoco_data.qfrc_applied[joint_map.dof_idx] = tau_applied;
        }
        
        // === Step 5: Physics simulation (1ms) ===
        unsafe { mj_step(self.mujoco_model, self.mujoco_data); }
        
        // === Step 6: Telemetry (optional) ===
        self.step_count += 1;
        SimStepOutput {
            t: (self.step_count as f32) * dt,
            pose: self.get_pose(),
            neural_activity: self.neuron_engine.get_activity_summary(),
            muscle_state: self.atlasov_state.get_summary(),
        }
    }
}
```

## Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
taas-kernels = { path = "../taas-kernels" }
taas-core = { path = "../taas-core" }
mujoco-sys = "0.2"  # C FFI bindings to MuJoCo
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"  # for benches
```

