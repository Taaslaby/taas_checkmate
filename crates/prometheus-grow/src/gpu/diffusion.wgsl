// -------------------------------------------------------------------------
// Filename:    TurboEuler
// Class:
// Purpose:     explicit Euler Laplacian stencil compute shader
// Description:
//
//    смотри как такая маленкая математика умеет сотворят мини миры
//
//    One workgroup processes a 4×4×4 tile of the interior grid.
//    Each thread writes one output cell.
//
//   c_new = c + dt * D * (sum_neighbors − 6*c) / dx² − dt * deg * c
//   result = clamp(c_new, lo, hi)
//
//     Boundaries are handled on the CPU side (only 6 faces, negligible cost).
//     References:
//     - Harris (2005) on Fast Fluid Dynamics Simulation on the GPU
//     - Owens et al. (2007) on A Survey of General-Purpose Computation on Graphics Hardware
//     - WebGPU Shading Language (WGSL) Specification (W3C Working Group)

//
// author: Atlasov anton
// -------------------------------------------------------------------------

struct Uniforms {
    nx: u32,
    ny: u32,
    nz: u32,
    pad0: u32,   // (alignment padding)
    dt: f32,
    diffusion: f32,
    degradation: f32,
    dx2: f32,
    lo: f32,
    hi: f32,
}

@group(0) @binding(0) var<storage, read>  field_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> field_out: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Interior only: skip boundary cells.
    let i = gid.x + 1u;
    let j = gid.y + 1u;
    let k = gid.z + 1u;

    if i >= u.nx - 1u || j >= u.ny - 1u || k >= u.nz - 1u {
        return;
    }

    let idx = (i * u.ny + j) * u.nz + k;

    // Read center and 6 neighbors.
    let c   = field_in[idx];
    let c_im = field_in[((i - 1u) * u.ny + j) * u.nz + k];
    let c_ip = field_in[((i + 1u) * u.ny + j) * u.nz + k];
    let c_jm = field_in[(i * u.ny + (j - 1u)) * u.nz + k];
    let c_jp = field_in[(i * u.ny + (j + 1u)) * u.nz + k];
    let c_km = field_in[(i * u.ny + j) * u.nz + (k - 1u)];
    let c_kp = field_in[(i * u.ny + j) * u.nz + (k + 1u)];

    let lap = (c_im + c_ip + c_jm + c_jp + c_km + c_kp) - 6.0 * c;
    let c_new = c + u.dt * u.diffusion * lap / u.dx2 - u.dt * u.degradation * c;
    field_out[idx] = clamp(c_new, u.lo, u.hi);
}
