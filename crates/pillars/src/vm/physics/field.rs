use wide::f32x8;

/// ███ Cool Physics: Ephaptic Field Coupling █████████████████████████████████
///
/// Neurons are coupled through the extracellular electric field.
/// When many neurons fire synchronously, they create an extracellular field
/// that can influence nearby neurons.
///
///   V_field = α * V_population_average
///   I_eph = g_eph * (V - V_field)
///
/// Registers:
///   field_curr (out): Ephaptic coupling current
///   v_reg (in): Membrane potential of this neuron
///   v_avg_reg (in): Population average voltage (pre-computed)
///   coupling (const): Field coupling strength α (typically 0.01-0.1)
///   g_eph (const): Ephaptic conductance
#[inline(always)]
pub fn op_phys_ephaptic_field(
    regs: &mut [f32x8; 16],
    field_curr: u8,
    v_reg: u8,
    v_avg_reg: u8,
    coupling: f32,
    g_eph: f32,
) {
    let v = regs[v_reg as usize];
    let v_avg = regs[v_avg_reg as usize];
    let alpha = f32x8::splat(coupling);
    let g = f32x8::splat(g_eph);

    // V_field = α * V_population_average
    let v_field = alpha * v_avg;
    // I = g_eph * (V - V_field)
    regs[field_curr as usize] = g * (v - v_field);
}

/// ███ Cool Physics: Ion Drift-Diffusion █████████████████████████████████████
///
/// Movement of ions (K⁺, Na⁺) through extracellular space under
/// electric field and concentration gradients (Nernst-Planck).
///
///   J = -D * dC/dx - mobility * C * dV/dx  (flux)
///   dC/dt = -dJ/dx (continuity)
///
/// Simplified as:
///   C_new = C + (D * Laplacian + mobility * E_field * gradient) * dt
///
/// Registers:
///   c_center (in/out): Concentration at this point
///   c_left (in): Concentration at left neighbor
///   c_right (in): Concentration at right neighbor
///   e_field (in): Electric field strength
///   diff_const (const): Diffusion coefficient
///   mobility (const): Ion mobility (related to diffusion via Einstein relation)
///   dx (const): Spatial step size
///   dt (const): Time step
#[inline(always)]
pub fn op_phys_ion_drift(
    regs: &mut [f32x8; 16],
    c_center: u8,
    c_left: u8,
    c_right: u8,
    e_field: u8,
    diff_const: f32,
    mobility: f32,
    dx: f32,
    dt: f32,
) {
    let cc = regs[c_center as usize];
    let cl = regs[c_left as usize];
    let cr = regs[c_right as usize];
    let ef = regs[e_field as usize];
    let D = f32x8::splat(diff_const);
    let mu = f32x8::splat(mobility);
    let dx_v = f32x8::splat(dx.max(1e-6));
    let dx2 = dx_v * dx_v;

    // Laplacian: (C_left + C_right - 2*C_center) / dx²
    let laplacian = (cl + cr - f32x8::splat(2.0) * cc) / dx2;

    // Gradient: (C_right - C_left) / (2 * dx)
    let gradient = (cr - cl) / (f32x8::splat(2.0) * dx_v);

    // dC/dt = D * Laplacian - mobility * E_field * gradient
    let transport = D * laplacian - mu * ef * gradient;
    regs[c_center as usize] = cc + transport * f32x8::splat(dt);
}

/// ███ Cool Physics: Reaction-Diffusion System ███████████████████████████████
///
/// A general reaction-diffusion equation for morphogens or ions.
/// Supports FitzHugh-Nagumo type dynamics:
///
///   du/dt = D * ∇²u + u - u³/3 - v + I
///   dv/dt = ε * (u + β - γ*v)
///
/// Registers:
///   u_center (in/out): Activator concentration
///   v_state (in/out): Inhibitor/refractory variable
///   u_left (in): Activator at left neighbor
///   u_right (in): Activator at right neighbor
///   D (const): Diffusion coefficient
///   epsilon (const): Recovery rate (slow manifold)
///   beta (const): Offset
///   gamma (const): Damping
///   I_reg (in): Input current
///   dx (const): Spatial step
///   dt (const): Time step
#[inline(always)]
pub fn op_phys_reaction_diffusion(
    regs: &mut [f32x8; 16],
    u_center: u8,
    v_state: u8,
    u_left: u8,
    u_right: u8,
    D: f32,
    epsilon: f32,
    beta: f32,
    gamma: f32,
    I_reg: u8,
    dx: f32,
    dt: f32,
) {
    let u = regs[u_center as usize];
    let v = regs[v_state as usize];
    let ul = regs[u_left as usize];
    let ur = regs[u_right as usize];
    let i_in = regs[I_reg as usize];
    let dx_v = f32x8::splat(dx.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // Laplacian: (ul + ur - 2*u) / dx²
    let laplacian = (ul + ur - f32x8::splat(2.0) * u) / (dx_v * dx_v);

    // FN kinetics: du/dt = D*laplacian + u - u³/3 - v + I
    let du = f32x8::splat(D) * laplacian + u - u * u * u / f32x8::splat(3.0) - v + i_in;
    let u_new = u + du * dt_v;

    // dv/dt = ε * (u + β - γ*v)
    let dv = f32x8::splat(epsilon) * (u + f32x8::splat(beta) - f32x8::splat(gamma) * v);
    let v_new = v + dv * dt_v;

    regs[u_center as usize] = u_new;
    regs[v_state as usize] = v_new;
}
