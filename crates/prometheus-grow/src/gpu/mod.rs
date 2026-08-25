//! wgpu compute acceleration.
//! GpuContext owns Device+Queue; pipelines take them as parameters.

use crate::substance::Grid;

#[cfg(feature = "gpu")]
#[path = "diffusion.rs"]
mod diffusion_impl;
#[cfg(feature = "gpu")]
pub mod trp_backend;

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    diffusion_pipeline: diffusion_impl::DiffusionPipeline,
    pub trp_pipeline: trp_backend::TrpPipeline,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("No wgpu adapter found")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("PROMETHEUS GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Device request failed: {e}"))?;

        let diffusion_pipeline = diffusion_impl::DiffusionPipeline::new(&device);
        let trp_pipeline = trp_backend::TrpPipeline::new(&device);

        Ok(GpuContext {
            device,
            queue,
            diffusion_pipeline,
            trp_pipeline,
        })
    }

    pub fn diffuse(
        &self,
        grid: &Grid,
        field: &[f64],
        c2: &mut [f64],
        dt: f64,
        diff: f64,
        deg: f64,
        lo: f64,
        hi: f64,
    ) {
        self.diffusion_pipeline.run(
            &self.device,
            &self.queue,
            grid,
            field,
            c2,
            dt,
            diff,
            deg,
            lo,
            hi,
        );
    }
}

/// Diffusion backend: CPU or GPU.
pub enum DiffusionBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(GpuContext),
}

impl DiffusionBackend {
    pub async fn new() -> Self {
        #[cfg(feature = "gpu")]
        {
            match GpuContext::new().await {
                Ok(ctx) => return DiffusionBackend::Gpu(ctx),
                Err(e) => eprintln!("[prometheus-grow] wgpu init failed: {e}, falling back to CPU"),
            }
        }
        DiffusionBackend::Cpu
    }

    pub fn cpu() -> Self {
        DiffusionBackend::Cpu
    }

    pub fn diffuse(
        &self,
        grid: &Grid,
        field: &[f64],
        c2: &mut [f64],
        dt: f64,
        diff: f64,
        deg: f64,
        lo: f64,
        hi: f64,
    ) {
        match self {
            DiffusionBackend::Cpu => diffuse_cpu(grid, field, c2, dt, diff, deg, lo, hi),
            #[cfg(feature = "gpu")]
            DiffusionBackend::Gpu(ctx) => ctx.diffuse(grid, field, c2, dt, diff, deg, lo, hi),
        }
    }
}

fn diffuse_cpu(
    grid: &Grid,
    field: &[f64],
    c2: &mut [f64],
    dt: f64,
    diff: f64,
    deg: f64,
    lo: f64,
    hi: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx2 = grid.cell * grid.cell;
    let inner_n = (nx.saturating_sub(2)) * (ny.saturating_sub(2)) * (nz.saturating_sub(2));
    if inner_n == 0 {
        return;
    }

    use rayon::prelude::*;
    let vals: Vec<f64> = (0..inner_n)
        .into_par_iter()
        .map(|lin| {
            let nz2 = nz - 2;
            let ny2 = ny - 2;
            let i = lin / (ny2 * nz2) + 1;
            let r = lin % (ny2 * nz2);
            let j = r / nz2 + 1;
            let k = r % nz2 + 1;
            let idx = grid.flat(i, j, k);
            let lap = (field[grid.flat(i - 1, j, k)]
                + field[grid.flat(i + 1, j, k)]
                + field[grid.flat(i, j - 1, k)]
                + field[grid.flat(i, j + 1, k)]
                + field[grid.flat(i, j, k - 1)]
                + field[grid.flat(i, j, k + 1)]
                - 6.0 * field[idx])
                / dx2;
            (field[idx] + dt * diff * lap - dt * deg * field[idx]).clamp(lo, hi)
        })
        .collect();
    let mut lin = 0usize;
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in 1..nz.saturating_sub(1) {
                c2[grid.flat(i, j, k)] = vals[lin];
                lin += 1;
            }
        }
    }
}

pub fn interior_index(coord: usize, n: usize) -> Option<usize> {
    if n < 2 {
        return None;
    }
    match coord {
        0 => Some(1),
        c if c == n - 1 => Some(n - 2),
        _ => Some(coord),
    }
}
