//! TRP attention on wgpu: T·R^T pairwise matmul.
//! Compile once, dispatch many times — &Device + &Queue as parameters.

use crate::trp::{TrpEmbedding, TrpParams, TRP_DIM};
use crate::Vec3;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTrpUniforms {
    n: u32,
    d: u32,
    pad0: u32,
    pad1: u32,
    sqrt_d: f32,
    inv_2sigma2: f32,
    prune_threshold: f32,
    max_dist: f32,
}

pub struct TrpPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl TrpPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("trp_attention.wgsl"));

        let make_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trp BGL"),
            entries: &[
                make_entry(0, true),  // tropisms
                make_entry(1, true),  // receptors
                make_entry(2, true),  // positions
                make_entry(3, false), // synapses_out
                make_entry(4, false), // synapse_count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trp layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trp attention pipeline"),
            layout: Some(&layout),
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
        ids: &[usize],
        embeddings: &HashMap<usize, TrpEmbedding>,
        positions: &HashMap<usize, Vec3>,
        params: &TrpParams,
    ) -> Vec<(usize, usize, f32)> {
        let n = ids.len();
        if n == 0 {
            return vec![];
        }
        let d = TRP_DIM;

        // Pack data.
        let mut tropisms = vec![0.0f32; n * d];
        let mut receptors = vec![0.0f32; n * d];
        let mut pos_flat = vec![0.0f32; n * 3];
        for (local, &gid) in ids.iter().enumerate() {
            let emb = embeddings.get(&gid).cloned().unwrap_or(TrpEmbedding {
                receptor: [0.0; TRP_DIM],
                tropism: [0.0; TRP_DIM],
                troph: 0.0,
            });
            for k in 0..d {
                tropisms[local * d + k] = emb.tropism[k];
                receptors[local * d + k] = emb.receptor[k];
            }
            let p = positions.get(&gid).copied().unwrap_or(Vec3::ZERO);
            pos_flat[local * 3] = p.x as f32;
            pos_flat[local * 3 + 1] = p.y as f32;
            pos_flat[local * 3 + 2] = p.z as f32;
        }

        let t_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trp T"),
            contents: bytemuck::cast_slice(&tropisms),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let r_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trp R"),
            contents: bytemuck::cast_slice(&receptors),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trp P"),
            contents: bytemuck::cast_slice(&pos_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let max_syns = (n as u64 * 64).min(16_000_000);
        let syn_size = (max_syns * 3) as wgpu::BufferAddress * 4;
        let syn_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trp syn out"),
            size: syn_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cnt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trp cnt"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let sigma2 = params.dist_sigma * params.dist_sigma;
        let unis = GpuTrpUniforms {
            n: n as u32,
            d: d as u32,
            pad0: 0,
            pad1: 0,
            sqrt_d: 1.0 / params.sqrt_d,
            inv_2sigma2: if sigma2 > 1e-12 {
                (0.5 / sigma2) as f32
            } else {
                1000.0
            },
            prune_threshold: params.prune_threshold,
            max_dist: params.max_dist as f32,
        };
        let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trp unis"),
            contents: bytemuck::bytes_of(&unis),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trp BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: syn_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cnt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uni_buf.as_entire_binding(),
                },
            ],
        });

        let wx = ((n as u32).div_ceil(16)).max(1);
        let wy = ((n as u32).div_ceil(16)).max(1);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trp enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trp pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wx, wy, 1);
        }

        let staging_syn = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trp staging syn"),
            size: syn_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging_cnt = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trp staging cnt"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&syn_buf, 0, &staging_syn, 0, syn_size);
        enc.copy_buffer_to_buffer(&cnt_buf, 0, &staging_cnt, 0, 4);
        queue.submit(Some(enc.finish()));

        // Read count.
        let (tx, rx) = std::sync::mpsc::channel();
        staging_cnt
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let cnt_data = staging_cnt.slice(..).get_mapped_range();
        let syn_count =
            u32::from_ne_bytes([cnt_data[0], cnt_data[1], cnt_data[2], cnt_data[3]]) as usize;
        drop(cnt_data);
        staging_cnt.unmap();

        if syn_count == 0 {
            return vec![];
        }

        // Read synapses.
        let (tx, rx) = std::sync::mpsc::channel();
        staging_syn
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let syn_data = staging_syn.slice(..).get_mapped_range();
        let syns_f32: &[f32] = bytemuck::cast_slice(&syn_data);

        let count = syn_count.min(max_syns as usize);
        let mut result = Vec::with_capacity(count);
        let id_map = ids.to_vec();
        for s in 0..count {
            let pre = syns_f32[s * 3] as usize;
            let post = syns_f32[s * 3 + 1] as usize;
            let score = syns_f32[s * 3 + 2];
            if pre < id_map.len() && post < id_map.len() {
                result.push((id_map[pre], id_map[post], score));
            }
        }
        drop(syn_data);
        staging_syn.unmap();
        result
    }
}
