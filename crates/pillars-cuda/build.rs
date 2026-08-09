//! build.rs — two GPU layers for pillars-cuda.
//!
//! 1. **NVRTC (JIT)**: `#[dual_neuron]` generates CUDA C++ as a string at
//!    compile time; the NVRTC runtime library compiles it to PTX on the fly.
//!    We only link against `nvrtc.lib` + `cuda.lib` (driver API) — no .cu
//!    files are involved for the zoo models.
//! 2. **AOT microkernels**: `kernels/*.cu` (the `gpu` module — GpuContext,
//!    CSR SpMV, STDP, T3, decoders, …) are compiled with nvcc into a static
//!    lib. Each .cu is one focused microkernel (pillars style).

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // ═══════════════════════════════════════════════════════════
    //  Part 1 — NVRTC runtime (zoo models, JIT)
    // ═══════════════════════════════════════════════════════════
    let cuda = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_PATH_V13_3"))
        .ok()
        .map(PathBuf::from)
        .or_else(detect_from_standard_locations)
        .unwrap_or_else(|| {
            eprintln!("cargo:warning=pillars-cuda: CUDA_PATH not set — build will still succeed, but the crate is unusable at runtime without CUDA.");
            PathBuf::from("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
        });

    let lib = cuda.join("lib").join("x64");
    if lib.exists() {
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=dylib=nvrtc");
        println!("cargo:rustc-link-lib=dylib=cuda");
    } else {
        eprintln!(
            "cargo:warning=pillars-cuda: CUDA lib dir not found at {}",
            lib.display()
        );
    }

    // Make nvrtc64_*.dll findable when running tests / examples.
    let bin = cuda.join("bin");
    if bin.exists() {
        if let Ok(path) = std::env::var("PATH") {
            if !path
                .split(';')
                .any(|p| p == bin.to_str().unwrap_or_default())
            {
                println!("cargo:rustc-env=PATH={};{}", bin.display(), path);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Part 2 — AOT microkernels (gpu module, nvcc)
    // ═══════════════════════════════════════════════════════════
    compile_aot_microkernels();

    println!("cargo:rerun-if-changed=build.rs");
}

/// Compile `kernels/*.cu` (one microkernel per file) into a static lib with nvcc.
///
/// Graceful fallback: if nvcc (or the MSVC toolchain on Windows) is not
/// available, the microkernels are skipped — every GPU call then returns an
/// error at runtime. This keeps `cargo check --workspace` green on machines
/// without a CUDA toolkit.
fn compile_aot_microkernels() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // AOT microkernels to compile (one .cu per concern — GpuContext infra,
    // neuron engines, CSR propagation, learning rules, T3/BPTT, decoders).
    let kernels = [
        "biophys.cu",
        "bptt.cu",
        "csr_spmv_tanh.cu",
        "decoders.cu",
        "edge_predictor.cu",
        "fused.cu",
        "gates.cu",
        "gco.cu",
        "gmm_sampler.cu",
        "gpu_context.cu",
        "hh.cu",
        "interference.cu",
        "izhikevich.cu",
        "kuramoto.cu",
        "snn_kernels.cu",
        "stdp.cu",
        "stdp_dopamine.cu",
        "t3_bptt_jacobian.cu",
        "t3_bptt_step.cu",
        "t3_fused_forward_bptt.cu",
        "t3_update.cu",
        "topk_select.cu",
        "xor_attention.cu",
    ];

    // Graceful fallback: if nvcc is not available (no CUDA toolkit), skip GPU
    // kernels entirely instead of panicking. The crate still compiles — the
    // FFI stubs simply never get linked, and every GPU call returns an error
    // at runtime.
    let Some(nvcc) = which_nvcc() else {
        eprintln!(
            "cargo:warning=pillars-cuda: nvcc not found — SKIPPING GPU microkernel compilation. \
             Install the CUDA toolkit (or set CUDA_HOME/CUDA_PATH) to enable GPU acceleration."
        );
        return;
    };

    // On Windows nvcc needs the MSVC toolchain (cl.exe + INCLUDE/LIB) even
    // though cl.exe is not on PATH. Source vcvars64.bat once and re-apply its
    // environment to every nvcc/lib invocation.
    let msvc_env = msvc_env();
    if cfg!(windows) && msvc_env.is_none() {
        eprintln!(
            "cargo:warning=pillars-cuda: MSVC toolchain (vcvars64.bat) not found — SKIPPING GPU microkernel compilation. \
             Install Visual Studio with the C++ workload to enable GPU acceleration."
        );
        return;
    }

    let mut obj_files = Vec::new();

    // Auto-detect GPU compute capability
    let arch = detect_arch();

    for &kernel in &kernels {
        let src = kernels_dir.join(kernel);
        let obj = out_dir.join(kernel.replace(".cu", obj_ext()));

        println!("cargo:rerun-if-changed={}", src.display());

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg(&arch)
            .arg("-O3")
            .arg("--use_fast_math")
            // M_PI is a POSIX constant: MSVC's math.h only defines it when
            // _USE_MATH_DEFINES is set (harmless no-op on Linux/glibc).
            .arg("-D_USE_MATH_DEFINES");
        if !cfg!(windows) {
            cmd.arg("-Xcompiler=-fPIC");
        }
        apply_msvc_env(&mut cmd, &msvc_env);

        let status = cmd.status().expect("nvcc failed");
        if !status.success() {
            panic!("nvcc compilation failed for {}", kernel);
        }
        obj_files.push(obj);
    }

    // Bundle into a static library: ar on Unix, lib.exe on Windows.
    let lib_name = if cfg!(windows) {
        "cuda_kernels.lib"
    } else {
        "libcuda_kernels.a"
    };
    let lib_path = out_dir.join(lib_name);
    if cfg!(windows) {
        let mut lib_cmd = Command::new("lib");
        lib_cmd.arg(format!("/OUT:{}", lib_path.display()));
        for obj in &obj_files {
            lib_cmd.arg(obj);
        }
        apply_msvc_env(&mut lib_cmd, &msvc_env);
        let status = lib_cmd.status().expect("lib failed");
        if !status.success() {
            panic!("lib.exe failed to create static library");
        }
    } else {
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("rcs").arg(&lib_path);
        for obj in &obj_files {
            ar_cmd.arg(obj);
        }
        let status = ar_cmd.status().expect("ar failed");
        if !status.success() {
            panic!("ar failed to create static library");
        }
    }

    // Link CUDA runtime + C++ stdlib (stdc++ is a GCC/Clang lib; MSVC uses
    // the CRT that rust links automatically).
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    if !cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if let Some(cuda_home) = get_cuda_home() {
        if cfg!(windows) {
            println!("cargo:rustc-link-search={}\\lib\\x64", cuda_home);
        } else {
            println!("cargo:rustc-link-search={}/lib64", cuda_home);
            println!("cargo:rustc-link-search={}/lib", cuda_home);
        }
    }
}

fn obj_ext() -> &'static str {
    if cfg!(windows) {
        ".obj"
    } else {
        ".o"
    }
}

fn apply_msvc_env(cmd: &mut Command, env_map: &Option<HashMap<String, String>>) {
    if let Some(map) = env_map {
        for (k, v) in map {
            cmd.env(k, v);
        }
    }
}

/// Locate the MSVC dev environment by sourcing vcvars64.bat (via vswhere) and
/// capturing the resulting environment variables (PATH/INCLUDE/LIB/…). Returns
/// `None` when VS with the C++ workload isn't installed.
#[cfg(windows)]
fn msvc_env() -> Option<HashMap<String, String>> {
    let vswhere =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    let vs_path = if vswhere.exists() {
        let out = Command::new(&vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
            .ok()?;
        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if p.is_empty() {
            return None;
        }
        PathBuf::from(p)
    } else {
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community")
    };

    let vcvars = vs_path.join("VC\\Auxiliary\\Build\\vcvars64.bat");
    if !vcvars.exists() {
        return None;
    }

    // Source vcvars via a temp .bat — inline `cmd /c "call … && set"` gets
    // mangled by cmd's quote handling, so a file is more robust.
    let bat = env::temp_dir().join("pillars_cuda_vcvars_env.bat");
    std::fs::write(
        &bat,
        format!("@echo off\ncall \"{}\" >nul 2>&1\nset\n", vcvars.display()),
    )
    .ok()?;
    let out = Command::new("cmd").arg("/c").arg(&bat).output().ok()?;
    let _ = std::fs::remove_file(&bat);

    let text = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    Some(map)
}

#[cfg(not(windows))]
fn msvc_env() -> Option<HashMap<String, String>> {
    None
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

fn which_nvcc() -> Option<String> {
    let candidates = [
        "nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/cuda/bin/nvcc",
    ];
    for &cmd in &candidates {
        if Command::new(cmd).arg("--version").output().is_ok() {
            return Some(cmd.to_string());
        }
    }
    if let Some(home) = get_cuda_home() {
        let path = format!("{}/bin/nvcc", home);
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

fn get_cuda_home() -> Option<String> {
    env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok()
}

fn detect_from_standard_locations() -> Option<PathBuf> {
    let base = PathBuf::from("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    let entries = std::fs::read_dir(&base).ok()?;
    let mut versions: Vec<(u32, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let v = name
                .strip_prefix('v')
                .and_then(|s| s.split('.').next().and_then(|n| n.parse::<u32>().ok()))?;
            Some((v, e.path()))
        })
        .collect();
    versions.sort_by_key(|(v, _)| *v);
    versions.last().map(|(_, p)| p.clone())
}
