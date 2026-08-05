//! build.rs — locate the CUDA toolkit and link NVRTC + driver API.
//!
//! `#[dual_neuron]` generates CUDA C++ as a string at compile time; the
//! NVRTC runtime library compiles it to PTX on the fly. We only need to
//! link against `nvrtc.lib` (NVRTC) and `cuda.lib` (driver API) — no
//! .cu files are ever compiled ahead-of-time.

use std::path::PathBuf;

fn main() {
    // Resolve the CUDA toolkit root: env var first, then the standard
    // install location for the detected major version.
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

    println!("cargo:rerun-if-changed=build.rs");
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
