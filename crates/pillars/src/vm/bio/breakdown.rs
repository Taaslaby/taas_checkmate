use wide::{f32x8, CmpGt};

/// Channel Breakdown (Пробой каналов)
/// If Voltage or Temperature exceeds critical thresholds, the membrane ruptures,
/// and conductance skyrockets to massive levels, causing catastrophic depolarization.
#[inline(always)]
pub fn op_bio_channel_breakdown(regs: &mut [f32x8; 16], g_reg: u8, v_reg: u8, t_reg: u8, v_crit: f32, t_crit: f32) {
    let g = regs[g_reg as usize];
    let v = regs[v_reg as usize];
    let t = regs[t_reg as usize];
    
    let vc = f32x8::splat(v_crit);
    let tc = f32x8::splat(t_crit);
    
    // Mask if V > V_crit OR T > T_crit (absolute voltage for dielectric breakdown)
    let mask_v = v.abs().cmp_gt(vc);
    let mask_t = t.cmp_gt(tc);
    let mask = mask_v | mask_t;
    
    // If breakdown occurs, conductance becomes catastrophically large
    let g_breakdown = f32x8::splat(1_000_000.0);
    
    regs[g_reg as usize] = mask.blend(g_breakdown, g);
}
