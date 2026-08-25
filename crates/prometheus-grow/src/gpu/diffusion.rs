//! wgpu compute pipeline for 7-point Laplacian stencil.
//! DiffusionPipeline: compile once, run many times.

use crate::substance::Grid;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUniforms {
    nx: u32,
    ny: u32,
    nz: u32,
    pad0: u32,
    dt: f32,
    diffusion: f32,
    degradation: f32,
    dx2: f32,
    lo: f32,
    hi: f32,
}

pub struct DiffusionPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl DiffusionPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("diffusion.wgsl"));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diffusion BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diffusion layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("diffusion pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &Grid,
        field: &[f64],
        field_out: &mut [f64],
        dt: f64,
        diff: f64,
        deg: f64,
        lo: f64,
        hi: f64,
    ) {
        let vol = grid.volume();
        assert_eq!(field.len(), vol);
        assert_eq!(field_out.len(), vol);

        let f32_in: Vec<f32> = field.iter().map(|&v| v as f32).collect();
        let buf_size = (vol * 4) as wgpu::BufferAddress;

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff in"),
            contents: bytemuck::cast_slice(&f32_in),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diff out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dx2 = grid.cell * grid.cell;
        let unis = GpuUniforms {
            nx: grid.nx as u32,
            ny: grid.ny as u32,
            nz: grid.nz as u32,
            pad0: 0,
            dt: dt as f32,
            diffusion: diff as f32,
            degradation: deg as f32,
            dx2: dx2 as f32,
            lo: lo as f32,
            hi: hi as f32,
        };
        let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff unis"),
            contents: bytemuck::bytes_of(&unis),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diff BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uni_buf.as_entire_binding(),
                },
            ],
        });

        let dx = ((grid.nx.max(2) - 2) as u32).div_ceil(4);
        let dy = ((grid.ny.max(2) - 2) as u32).div_ceil(4);
        let dz = ((grid.nz.max(2) - 2) as u32).div_ceil(4);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("diff enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diff pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dx, dy, dz);
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diff staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, buf_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let f32_out: &[f32] = bytemuck::cast_slice(&data);
        // The shader writes only interior cells; boundary cells must keep
        // their input values (the CPU backend leaves them untouched too —
        // `World::diffuse_substances` applies the boundary condition after).
        for (i, &v) in f32_out.iter().enumerate() {
            let k = i % grid.nz;
            let j = (i / grid.nz) % grid.ny;
            let x = i / (grid.nz * grid.ny);
            let is_boundary = x == 0
                || j == 0
                || k == 0
                || x == grid.nx - 1
                || j == grid.ny - 1
                || k == grid.nz - 1;
            field_out[i] = if is_boundary { field[i] } else { v as f64 };
        }
        drop(data);
        staging.unmap();
    }
}
