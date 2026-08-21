// ═══════════════════════════════════════════════════════════════
//  Network
// ═══════════════════════════════════════════════════════════════

/// A single directed edge: `pre` (index in the source population), `post`
/// (index in the target population), `weight` and axonal `delay` (steps,
/// `0` = delivered on the next step).
///
/// See [`Network::connect_edges`] for ergonomic connection.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub pre: u32,
    pub post: u32,
    pub weight: f32,
    pub delay: u8,
    /// Synapse type: `0` = static current (default), `1..` = a kinetic
    /// profile id from [`Network::register_kinetics`].
    pub syn_type: u8,
    /// Short-term plasticity profile: `0` = none (default), `1..` = a profile
    /// id from [`Network::register_plasticity`].
    pub plast_type: u8,
}

impl Edge {
    #[inline]
    pub fn new(pre: u32, post: u32, weight: f32, delay: u8) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type: 0,
            plast_type: 0,
        }
    }

    /// An edge with a kinetic synapse type (see [`Network::register_kinetics`]).
    #[inline]
    pub fn with_synapse(pre: u32, post: u32, weight: f32, delay: u8, syn_type: u8) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type,
            plast_type: 0,
        }
    }

    /// An edge with a kinetic synapse type **and** a short-term plasticity
    /// profile (see [`Network::register_kinetics`] / [`Network::register_plasticity`]).
    #[inline]
    pub fn with_synapse_plasticity(
        pre: u32,
        post: u32,
        weight: f32,
        delay: u8,
        syn_type: u8,
        plast_type: u8,
    ) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type,
            plast_type,
        }
    }
}
