use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // CUDA kernels to compile
    let kernels = [
        "xor_attention.cu",
        "izhikevich.cu",
        "interference.cu",
        "stdp.cu",
        "kuramoto.cu",
        "gates.cu",
        "decoders.cu",
        "gpu_context.cu",
        "fused.cu",
        "t3_update.cu",
        "t3_bptt_jacobian.cu",
        "t3_bptt_step.cu",
        "snn_kernels.cu",
        "gco.cu",
        "hh.cu",
        "csr_spmv_tanh.cu",
        "stdp_dopamine.cu",
        "t3_fused_forward_bptt.cu",
        "bptt.cu",
        "edge_predictor.cu",
        "gmm_sampler.cu",
        "topk_select.cu",
        "biophys.cu",
    ];
    let mut obj_files = Vec::new();
    let nvcc = which_nvcc();

    // Auto-detect GPU compute capability
    let arch = detect_arch();

    for &kernel in &kernels {
        let src = kernels_dir.join(kernel);
        let obj = out_dir.join(kernel.replace(".cu", ".o"));

        println!("cargo:rerun-if-changed={}", src.display());

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg(&arch)
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Xcompiler=-fPIC");

        let status = cmd.status().expect("nvcc failed");
        if !status.success() {
            panic!("nvcc compilation failed for {}", kernel);
        }
        obj_files.push(obj);
    }

    // Bundle into static library
    let lib_path = out_dir.join("libcuda_kernels.a");
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("ar failed");
    if !status.success() {
        panic!("ar failed to create static library");
    }

    // Link CUDA runtime + C++ stdlib
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    if let Some(cuda_home) = get_cuda_home() {
        println!("cargo:rustc-link-search={}/lib64", cuda_home);
        println!("cargo:rustc-link-search={}/lib", &cuda_home);
    }
}

fn detect_arch() -> String {
    // Try to detect via nvidia-smi
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if let Some((maj, min)) = s.split_once('.') {
            let arch = format!("-arch=sm_{}{}", maj.trim(), min.trim());
            eprintln!("cargo:warning=Detected GPU arch: {}", arch);
            return arch;
        }
    }

    // Try cuda runtime to detect
    if let Ok(out) = Command::new("nvcc").arg("--version").output() {
        let _s = String::from_utf8_lossy(&out.stdout);
        // Default to sm_89 for Ada (RTX 40xx) if nvcc supports it
        eprintln!("cargo:warning=nvcc found, defaulting to -arch=sm_89");
    }

    // Safe fallback: sm_75 (Turing) works on everything from 2018+
    eprintln!("cargo:warning=Could not detect GPU, falling back to -arch=sm_75");
    "-arch=sm_75".to_string()
}

fn which_nvcc() -> String {
    let candidates = [
        "nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/cuda/bin/nvcc",
    ];
    for &cmd in &candidates {
        if Command::new(cmd).arg("--version").output().is_ok() {
            return cmd.to_string();
        }
    }
    if let Some(home) = get_cuda_home() {
        let path = format!("{}/bin/nvcc", home);
        if std::path::Path::new(&path).exists() {
            return path;
        }
    }
    panic!("nvcc not found. Install CUDA toolkit or set CUDA_HOME");
}

fn get_cuda_home() -> Option<String> {
    env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok()
}
