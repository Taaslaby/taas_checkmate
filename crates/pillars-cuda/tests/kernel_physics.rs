//! Generic NVRTC launcher smoke test using a spring-damper physics kernel.

use pillars_cuda::{BufferArg, CudaKernel, LaunchConfig, ScalarArg};

const SPRING_SOURCE: &str = r#"
extern "C" __global__ void spring_step(
    float* x,
    float* velocity,
    const float* mass,
    const float* force,
    float stiffness,
    float damping,
    float dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acceleration = (force[i] - stiffness * x[i] - damping * velocity[i]) / mass[i];
    velocity[i] += acceleration * dt;
    x[i] += velocity[i] * dt;
}
"#;

#[test]
fn generic_kernel_runs_physics_not_neurons() {
    let kernel = CudaKernel::compile(SPRING_SOURCE, "spring_step")
        .expect("CUDA device/NVRTC required for generic kernel test");

    let mut x = vec![1.0f32; 257];
    let mut velocity = vec![0.0f32; 257];
    let mut mass = vec![1.0f32; 257];
    let mut force = vec![0.0f32; 257];
    let n = x.len() as i32;

    let mut buffers = [
        BufferArg::F32(&mut x),
        BufferArg::F32(&mut velocity),
        BufferArg::F32(&mut mass),
        BufferArg::F32(&mut force),
    ];
    kernel
        .launch(
            &mut buffers,
            &[
                ScalarArg::F32(4.0),
                ScalarArg::F32(0.2),
                ScalarArg::F32(0.01),
                ScalarArg::I32(n),
            ],
            LaunchConfig::for_len(n as usize).unwrap(),
        )
        .expect("spring kernel launch");

    assert!(x.iter().all(|value| value.is_finite() && *value < 1.0));
    assert!(velocity
        .iter()
        .all(|value| value.is_finite() && *value < 0.0));
}

#[test]
fn missing_kernel_entry_is_an_error() {
    let err = match CudaKernel::compile(SPRING_SOURCE, "spring_stepp") {
        Err(err) => err,
        Ok(_) => panic!("expected compile error for missing kernel entry"),
    };
    assert!(
        err.contains("not found") || err.contains("kernel entry"),
        "unexpected error: {err}"
    );
}

#[test]
fn suffix_entry_is_not_accepted() {
    // Regression: a request for `step` must NOT silently select `other_step`
    // via suffix matching. Exact `extern "C"` entry names only.
    const SOURCE: &str = r#"
extern "C" __global__ void other_step(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += 1.0f;
}
"#;
    let err = match CudaKernel::compile(SOURCE, "step") {
        Err(err) => err,
        Ok(_) => panic!("expected suffix `step` not to match `other_step`"),
    };
    assert!(err.contains("not found"), "unexpected error: {err}");

    // The exact entry still compiles and runs.
    let kernel = CudaKernel::compile(SOURCE, "other_step").expect("exact entry compiles");
    let mut x = vec![0.0f32; 64];
    let n = x.len() as i32;
    let mut buffers = [BufferArg::F32(&mut x)];
    kernel
        .launch(
            &mut buffers,
            &[ScalarArg::I32(n)],
            LaunchConfig::for_len(n as usize).unwrap(),
        )
        .expect("exact kernel launch");
    assert!(x.iter().all(|&value| value == 1.0));
}

#[test]
fn invalid_source_is_an_error() {
    let err = match CudaKernel::compile("this is not valid cuda", "k") {
        Err(err) => err,
        Ok(_) => panic!("expected compile error for invalid source"),
    };
    assert!(!err.is_empty(), "expected a compile error message");
}
