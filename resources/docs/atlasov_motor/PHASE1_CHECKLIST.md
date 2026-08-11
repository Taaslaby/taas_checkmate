# Phase 1 FlyWire Embodied Test — Implementation Checklist

## Overview
**Goal:** Verify that the full pipeline (neurons → muscles → MuJoCo physics) works end-to-end.
**Success Criteria:** Fly produces realistic motor patterns (tripod gait, recovery from perturbations).
**Timeline:** 2–3 days of focused work.

---

## Milestone 1: Create the embodied module (Day 1 morning)

> **Current layout:** this historical plan predates the workspace consolidation. Do not create `taas-embodied`, `taas-core`, or `taas-kernels` crates. New embodied code belongs in `pillars::api` / `pillars::zoo` and, when needed, `pillars-connectome`.

### M1.1: Workspace setup
- [ ] Extend the existing `pillars` embodied API, or create a new crate under `crates/` only when the feature needs a separate package. Do **not** run `cargo new --lib taas-embodied`.
- [ ] Use the existing workspace crates; do not add legacy members:
  ```toml
  members = [
      "crates/pillars", "crates/pillars-cuda", "crates/pillars-connectome",
      "crates/t3", "crates/train", "crates/taas-format",
  ]
  ```
- [ ] If a separate crate under `crates/` is justified, add dependencies relative to that crate's directory:
  ```toml
  [dependencies]
  pillars = { path = "../pillars" }
  pillars-connectome = { path = "../pillars-connectome" }
  mujoco-sys = "0.2"
  ndarray = "0.15"
  serde = { version = "1.0", features = ["derive"] }
  ```
  If the implementation belongs in the existing `crates/pillars` module, do not add a self-dependency; use `crate::api`, `crate::zoo`, and `crate::kernels` directly.

### M1.2: Core data structures
- [ ] Copy `fly_embodied_skeleton.rs` → `taas-embodied/src/fly_embodied.rs`
- [ ] Define `JointMapping`, `FlyPose`, `NeuralActivity`, `MuscleState`, `SimStepOutput`
- [ ] Stub out `FlyEmbodiedState` struct with all fields
- [ ] Export in `lib.rs`:
  ```rust
  pub mod fly_embodied;
  pub use fly_embodied::{FlyEmbodiedState, JointMapping, FlyPose};
  ```

### M1.3: Build test
- [ ] `cargo build -p taas-embodied` should succeed with no errors
- [ ] `cargo test --lib --no-run` should compile tests (even if they're all stubs)

---

## Milestone 2: Implement MuJoCo Binding Layer (Day 1 afternoon)

### M2.1: Safe MuJoCo wrapper
- [ ] Implement `FlyEmbodiedState::new()`:
  - [ ] Parse XML file path
  - [ ] Call `mj_loadXML()` and `mj_makeData()`
  - [ ] Validate model is not null
  - [ ] Store pointers safely (use `*mut` with clear ownership semantics)
  
- [ ] Implement helper functions:
  ```rust
  fn get_qpos(&self, dof: usize) -> f32
  fn get_qvel(&self, dof: usize) -> f32
  fn set_qfrc_applied(&mut self, dof: usize, tau: f32)
  ```

### M2.2: Physics step wrapper
- [ ] Implement `step_physics(&mut self)`:
  ```rust
  unsafe { mj_step(self.mujoco_model, self.mujoco_data); }
  ```
- [ ] Wrap in `Result<(), String>` for error handling

### M2.3: Data extraction
- [ ] Implement `get_pose()` stub:
  - [ ] Read body COM position from `mjData`
  - [ ] Read all joint angles from `qpos`
  - [ ] Return `FlyPose` struct
  
- [ ] Test: `cargo test -- --test-threads=1 test_pose_extraction`

---

## Milestone 3: Connect Neural Layer (Day 1 evening / Day 2 morning)

### M3.1: Motoneuron spike retrieval
- [ ] Implement `get_motoneuron_spikes()`:
  ```rust
  pub fn get_motoneuron_spikes(&self) -> Vec<i32> {
      self.motoneuron_indices.iter()
          .map(|&idx| self.neuron_engine.get_spike(idx))
          .collect()
  }
  ```

### M3.2: DelayBuffer integration
- [ ] In `step()`, push spikes into delay buffers:
  ```rust
  let motoneuron_spikes = self.get_motoneuron_spikes();
  let delayed_spikes: Vec<i32> = self.delay_buffers.iter_mut()
      .zip(motoneuron_spikes.iter())
      .map(|(buf, &spike)| buf.push(spike))
      .collect();
  ```

### M3.3: Test latency
- [ ] Write `test_nms_delay_latency()`:
  ```rust
  #[test]
  fn test_nms_delay_latency() {
      let mut fly = FlyEmbodiedState::new(...);
      for _ in 0..100 { fly.step().unwrap(); }
      // Inject spike at t=50ms
      // Check that muscle responds at t=51.5ms (1.5ms delay)
      assert!(delay_measured > 1.4 && delay_measured < 1.6);
  }
  ```

---

## Milestone 4: Connect Muscle Layer (Day 2)

### M4.1: Atlasov integration
- [ ] In `step()`, call `atlasov_state.step()`:
  ```rust
  self.atlasov_state.step(&delayed_spikes, self.dt_physics);
  let (forces, stiffness, damping) = self.atlasov_state.get_outputs();
  ```

### M4.2: Force application to MuJoCo
- [ ] Implement force loop in `step()`:
  ```rust
  for (joint_map, &df, &k, &b) in izip!(
      &self.joint_mappings,
      forces.iter(),
      stiffness.iter(),
      damping.iter(),
  ) {
      let q = self.get_qpos(joint_map.dof_idx);
      let q_dot = self.get_qvel(joint_map.dof_idx);
      let tau_applied = df - k * (q - joint_map.q_rest) - b * q_dot;
      self.set_qfrc_applied(joint_map.dof_idx, tau_applied);
  }
  ```

### M4.3: Test force application
- [ ] Write `test_force_application()`:
  ```rust
  #[test]
  fn test_force_application() {
      let mut fly = FlyEmbodiedState::new(...);
      fly.step().unwrap();
      
      // If atlasov generates ΔF > 0, joint should move
      let pose1 = fly.get_pose().unwrap();
      fly.step().unwrap();
      let pose2 = fly.get_pose().unwrap();
      
      assert!(pose2.joints[0] != pose1.joints[0], "Joint should move");
  }
  ```

---

## Milestone 5: Behavioral Tests (Day 2 afternoon / Day 3)

### M5.1: Smoke test (baseline)
- [ ] `test_embodied_initialization()`:
  - [ ] Create fly
  - [ ] Step 10 times
  - [ ] Check no panics, reasonable pose values
  
### M5.2: Spontaneous walking
- [ ] `test_spontaneous_walking_pattern()`:
  - [ ] Simulate 10 seconds (10,000 steps)
  - [ ] Record all leg joint angles
  - [ ] Compute alternation score (tripod: L-mid, R-front/back alternates)
  - [ ] Assert alternation_score > 0.6
  
### M5.3: Motor latency
- [ ] `test_motor_latency_end_to_end()`:
  - [ ] Inject 5nA current into DCN
  - [ ] Measure spike time (t_spike)
  - [ ] Measure leg movement start (t_move = when angle changes >0.05 rad)
  - [ ] Assert 1.5ms < (t_move - t_spike) < 2.5ms
  
### M5.4: Perturbation recovery
- [ ] `test_posture_stability_perturbation()`:
  - [ ] Walk for 2 seconds
  - [ ] Apply 50mN impulse to body
  - [ ] Check: fly doesn't fall for 5 more seconds
  - [ ] Assert uptime > 95%

---

## Milestone 6: Performance & Polish (Day 3)

### M6.1: Benchmark
- [ ] Measure throughput: steps/second on RTX 4060 Ti
- [ ] Target: >1000 FPS headless (10× real-time)
- [ ] Profile with `cargo bench -p taas-embodied`

### M6.2: Documentation
- [ ] Add module-level docs to `fly_embodied.rs`
- [ ] Document `JointMapping` struct (how to populate it)
- [ ] Document control flow in `step()` method
- [ ] Add example: `examples/embodied_fly_sim.rs`

### M6.3: CI integration
- [ ] Add to GitHub Actions (if using)
- [ ] Ensure all tests pass on push

---

## Testing Utilities (to implement)

```rust
// Utility: compute alternation score for tripod gait
fn compute_alternation(leg_angles: &[f32]) -> f32 {
    // Compare L-mid with R-front/back phases
    // Returns 0.0 (no alternation) to 1.0 (perfect alternation)
    todo!()
}

// Utility: detect when body is falling
fn is_falling(pose: &FlyPose) -> bool {
    pose.body_height < 0.8  // Threshold: below 80% nominal height
}

// Utility: detect joint movement
fn joint_moved(angle_before: f32, angle_after: f32, threshold: f32) -> bool {
    (angle_after - angle_before).abs() > threshold
}
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MuJoCo FFI crashes | Use `unsafe` blocks carefully, validate all pointers before deref |
| qfrc_applied not applied correctly | Start with zero forces, verify MuJoCo accepts them |
| DelayBuffer off-by-one | Unit test DelayBuffer separately before use |
| Unrealistic behavior | Compare with known good reference (e.g., simple LIF motor control) |
| Performance collapse | Profile early; check that loop doesn't allocate excessively |

---

## Success Criteria (Definition of Done)

✅ All 4 tests pass (smoke, walking, latency, perturbation)
✅ Throughput >1000 FPS on RTX 4060 Ti
✅ Documentation complete
✅ No unsafe undefined behavior (run with `MIRIFLAGS=-Zmiri-strict-provenance`)
✅ Code review by collaborator

---

## Next Steps After Phase 1

- Phase 2: Biological calibration (sweep K_d, τ_pump, match published twitch kinetics)
- Phase 3: GPU optimization (SIMD/CUDA benchmarks)
- Phase 4: Behavior exploration (navigation, obstacle avoidance, etc.)

> ⚠️ **SUPERSEDED:** этот план описывает старую структуру. `taas-embodied`, `taas-core` и `taas-kernels` больше не являются workspace crates. Реализацию продолжать через `pillars::api`, `pillars::zoo` и `pillars-connectome`; текущий workspace перечислен в `Cargo.toml`.
