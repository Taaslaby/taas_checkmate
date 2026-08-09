//! CSR spike propagation routines and queue operations.

use crate::types::SpikeQueue;
use pillars::kernels::csr::ValidatedCsrTopology;
use pillars::kernels::synapse::{ReceptorType, SynapticKinetics};

#[inline]
pub fn csr_propagate(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    spikes: &[i32],
    n_neurons: usize,
    I_buf: &mut [f32],
    scale: f32,
) -> u32 {
    let max_syn = col_indices.len().min(values.len());
    let mut total = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            total += 1;
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1] as usize).min(max_syn);
            let mult = if i < nt_types.len() {
                nt_types[i] * scale
            } else {
                scale
            };
            let mut j = start;
            while j + 4 <= end {
                let t0 = col_indices[j].max(0) as usize;
                let t1 = col_indices[j + 1].max(0) as usize;
                let t2 = col_indices[j + 2].max(0) as usize;
                let t3 = col_indices[j + 3].max(0) as usize;
                if t0 < I_buf.len() {
                    I_buf[t0] += values[j] * mult;
                }
                if t1 < I_buf.len() {
                    I_buf[t1] += values[j + 1] * mult;
                }
                if t2 < I_buf.len() {
                    I_buf[t2] += values[j + 2] * mult;
                }
                if t3 < I_buf.len() {
                    I_buf[t3] += values[j + 3] * mult;
                }
                j += 4;
            }
            for j in j..end {
                let col = col_indices[j].max(0) as usize;
                if col < I_buf.len() {
                    I_buf[col] += values[j] * mult;
                }
            }
        }
    }
    total
}

#[inline]
pub fn record_spikes(
    spikes: &[i32],
    n_neurons: usize,
    spike_buffer: &mut [i32],
    step: usize,
) -> u32 {
    let offset = step * n_neurons;
    let mut step_count = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            let idx = offset + i;
            if idx < spike_buffer.len() {
                spike_buffer[idx] = 1;
            }
            step_count += 1;
        }
    }
    step_count
}

#[inline]
pub fn push_spikes_to_queue(spikes: &[i32], n_neurons: usize, queue: &mut SpikeQueue) -> u32 {
    let mut count = 0u32;
    for i in 0..n_neurons.min(spikes.len()) {
        if spikes[i] != 0 {
            queue.push(i as u32);
            count += 1;
        }
    }
    count
}

#[inline]
pub fn csr_propagate_from_queue(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    queue: &mut SpikeQueue,
    I_buf: &mut [f32],
    scale: f32,
) -> u32 {
    let max_syn = col_indices.len().min(values.len());
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        total += 1;
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let mult = if i < nt_types.len() {
            nt_types[i] * scale
        } else {
            scale
        };
        let mut j = start;
        while j + 4 <= end {
            let t0 = col_indices[j].max(0) as usize;
            let t1 = col_indices[j + 1].max(0) as usize;
            let t2 = col_indices[j + 2].max(0) as usize;
            let t3 = col_indices[j + 3].max(0) as usize;
            if t0 < I_buf.len() {
                I_buf[t0] += values[j] * mult;
            }
            if t1 < I_buf.len() {
                I_buf[t1] += values[j + 1] * mult;
            }
            if t2 < I_buf.len() {
                I_buf[t2] += values[j + 2] * mult;
            }
            if t3 < I_buf.len() {
                I_buf[t3] += values[j + 3] * mult;
            }
            j += 4;
        }
        for j in j..end {
            let col = col_indices[j].max(0) as usize;
            if col < I_buf.len() {
                I_buf[col] += values[j] * mult;
            }
        }
    }
    total
}

/// Queue propagation through a topology validated by the owning runner.
///
/// Unlike [`csr_propagate_from_queue`], this path performs no CSR validation
/// or row-pointer clamping in the timestep loop.
#[inline]
pub fn csr_propagate_from_queue_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    nt_types: &[f32],
    queue: &mut SpikeQueue,
    I_buf: &mut [f32],
    scale: f32,
) -> u32 {
    assert_eq!(
        values.len(),
        topology.col_indices().len(),
        "csr_propagate_from_queue_with_topology: values length mismatch"
    );
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        if i >= topology.n_neurons() {
            continue;
        }
        total += 1;
        let mult = nt_types.get(i).copied().unwrap_or(1.0) * scale;
        let start = topology.row_ptr()[i] as usize;
        let end = topology.row_ptr()[i + 1] as usize;
        for j in start..end {
            let post = topology.col_indices()[j] as usize;
            if post < I_buf.len() {
                I_buf[post] += values[j] * mult;
            }
        }
    }
    total
}

/// Conductance propagation through a topology validated by the owning runner.
#[inline]
pub fn csr_propagate_conductance_from_queue_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    receptor_types: &[u8],
    queue: &mut SpikeQueue,
    kinetics: &mut SynapticKinetics,
    scale: f32,
) -> u32 {
    assert_eq!(
        values.len(),
        topology.col_indices().len(),
        "csr_propagate_conductance_with_topology: values length mismatch"
    );
    assert_eq!(
        receptor_types.len(),
        values.len(),
        "csr_propagate_conductance_with_topology: receptor_types length mismatch"
    );
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        if i >= topology.n_neurons() {
            continue;
        }
        total += 1;
        let start = topology.row_ptr()[i] as usize;
        let end = topology.row_ptr()[i + 1] as usize;
        for j in start..end {
            kinetics.jump(
                topology.col_indices()[j] as usize,
                ReceptorType::from_u8(receptor_types[j]),
                values[j] * scale,
            );
        }
    }
    total
}

#[inline]
pub fn csr_propagate_conductance_from_queue(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    receptor_types: &[u8],
    queue: &mut SpikeQueue,
    kinetics: &mut SynapticKinetics,
    scale: f32,
) -> u32 {
    // The conductance path requires one receptor type per synapse (built via
    // `ConnectomeData::build_receptor_types*`). An empty or short array means the
    // caller followed the instant-current route (`csr_propagate_from_queue`,
    // nt_types only) but landed here — fail loudly instead of silently dropping
    // every synapse (old behaviour: `max_syn = min(..)` collapsed to 0).
    assert_eq!(
        col_indices.len(),
        values.len(),
        "csr_propagate_conductance_from_queue: col_indices.len() ({}·) != values.len() ({}·)",
        col_indices.len(),
        values.len()
    );
    assert_eq!(
        col_indices.len(),
        receptor_types.len(),
        "csr_propagate_conductance_from_queue: receptor_types must be per-synapse \
         ({} entries, got {} — call ConnectomeData::build_receptor_types())",
        col_indices.len(),
        receptor_types.len()
    );
    let max_syn = col_indices.len();
    let mut total = 0u32;
    while let Some(id) = queue.pop() {
        let i = id as usize;
        total += 1;
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let mut j = start;
        while j + 4 <= end {
            let t0 = col_indices[j].max(0) as usize;
            let t1 = col_indices[j + 1].max(0) as usize;
            let t2 = col_indices[j + 2].max(0) as usize;
            let t3 = col_indices[j + 3].max(0) as usize;
            let w0 = values[j] * scale;
            let w1 = values[j + 1] * scale;
            let w2 = values[j + 2] * scale;
            let w3 = values[j + 3] * scale;
            let rec0 = ReceptorType::from_u8(receptor_types[j]);
            let rec1 = ReceptorType::from_u8(receptor_types[j + 1]);
            let rec2 = ReceptorType::from_u8(receptor_types[j + 2]);
            let rec3 = ReceptorType::from_u8(receptor_types[j + 3]);
            kinetics.jump(t0, rec0, w0);
            kinetics.jump(t1, rec1, w1);
            kinetics.jump(t2, rec2, w2);
            kinetics.jump(t3, rec3, w3);
            j += 4;
        }
        while j < end {
            let post = col_indices[j].max(0) as usize;
            let w = values[j] * scale;
            let rec = ReceptorType::from_u8(receptor_types[j]);
            kinetics.jump(post, rec, w);
            j += 1;
        }
    }
    total
}
