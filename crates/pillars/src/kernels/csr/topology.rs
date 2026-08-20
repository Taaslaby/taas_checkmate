use std::sync::Arc;

// ─── Safe CSR access helpers ─────────────────────────────

/// Safe CSR row pointer access: clamp index and validate bounds.
///
/// Внутри модуля не используется (функции индексируют `row_ptr.get()`
/// напрямую) — публичный API для `core::sparse` и python-ABI.
#[inline(always)]
pub fn bound_row(row_ptr: &[i32], idx: usize, n: usize) -> usize {
    if idx >= row_ptr.len() {
        return 0;
    }
    let r = row_ptr[idx];
    if r < 0 {
        0
    } else {
        r.min(n as i32) as usize
    }
}

/// Safe CSR column index access: validate column is within [0, n). Panics on invalid.
#[inline(always)]
pub fn bound_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() {
        return 0;
    }
    let c = col_indices[idx];
    debug_assert!(
        c >= 0 && (c as usize) < n,
        "bound_col: invalid column {} at idx {}, n={}, col_indices.len={}",
        c,
        idx,
        n,
        col_indices.len()
    );
    if c < 0 || c as usize >= n {
        panic!(
            "bound_col: out of range column {} at idx {}, n={}",
            c, idx, n
        );
    }
    c as usize
}

// ─── Validated CSR topology ──────────────────────────────

/// Borrowed CSR topology validated once for reuse across simulation steps.
///
/// The wrapper owns no buffers and therefore remains cheap to copy and safe to
/// share across rayon workers. Synaptic values/weights stay outside the wrapper
/// so mutable learning kernels can update them without violating the borrow
/// contract.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCsrTopology<'a> {
    pub(crate) row_ptr: &'a [i32],
    pub(crate) col_indices: &'a [i32],
    pub(crate) n_neurons: usize,
}

impl<'a> ValidatedCsrTopology<'a> {
    /// Validate and borrow a CSR topology. This is the O(edges) boundary check.
    pub fn new(row_ptr: &'a [i32], col_indices: &'a [i32], n_neurons: usize) -> Self {
        assert_eq!(
            row_ptr.len(),
            n_neurons
                .checked_add(1)
                .expect("CSR: neuron count overflows row_ptr length"),
            "CSR: row_ptr must have n + 1 entries"
        );
        assert_eq!(
            row_ptr.first().copied(),
            Some(0),
            "CSR: row_ptr must start at 0"
        );
        let mut previous = 0usize;
        for (i, &raw) in row_ptr.iter().enumerate() {
            assert!(raw >= 0, "CSR: row_ptr[{i}] is negative ({raw})");
            let offset = raw as usize;
            assert!(
                offset >= previous,
                "CSR: row_ptr must be monotonic at index {i} ({offset} < {previous})"
            );
            assert!(
                offset <= col_indices.len(),
                "CSR: row_ptr[{i}]={offset} exceeds {} synapses",
                col_indices.len()
            );
            previous = offset;
        }
        assert_eq!(
            previous,
            col_indices.len(),
            "CSR: final row_ptr offset must equal synapse count"
        );
        for (i, &raw) in col_indices.iter().enumerate() {
            assert!(
                raw >= 0 && (raw as usize) < n_neurons,
                "CSR: col_indices[{i}]={raw} out of range for {n_neurons} neurons"
            );
        }
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    #[inline(always)]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    #[inline(always)]
    pub fn row_ptr(&self) -> &'a [i32] {
        self.row_ptr
    }

    #[inline(always)]
    pub fn col_indices(&self) -> &'a [i32] {
        self.col_indices
    }

    /// Construct a view from an already validated owner without rescanning it.
    #[inline(always)]
    pub(crate) fn from_validated(
        row_ptr: &'a [i32],
        col_indices: &'a [i32],
        n_neurons: usize,
    ) -> Self {
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    #[inline(always)]
    pub(crate) fn bounds(&self, row: usize) -> (usize, usize) {
        assert!(row < self.n_neurons, "CSR: row {row} out of range");
        (self.row_ptr[row] as usize, self.row_ptr[row + 1] as usize)
    }
}

/// Owning, immutable CSR topology for long-lived simulation objects.
///
/// Construction performs the O(edges) validation once. `as_topology()` creates
/// a borrowed view in O(1), so a runner can reuse the validated structure on
/// every timestep without a self-referential struct or unsafe code. Synaptic
/// values remain outside this type because learning kernels mutate them.
#[derive(Clone, Debug)]
pub struct ValidatedCsrGraph {
    row_ptr: Arc<[i32]>,
    col_indices: Arc<[i32]>,
    n_neurons: usize,
}

impl ValidatedCsrGraph {
    /// Copy CSR topology into an owned validated snapshot.
    ///
    /// The copy is intentional: existing `ConnectomeData` keeps public mutable
    /// `Vec`s for ABI compatibility, while this snapshot remains immutable and
    /// safe to share across simulation steps and rayon workers.
    pub fn from_slices(row_ptr: &[i32], col_indices: &[i32], n_neurons: usize) -> Self {
        let row_ptr: Arc<[i32]> = Arc::from(row_ptr);
        let col_indices: Arc<[i32]> = Arc::from(col_indices);
        let _ = ValidatedCsrTopology::new(&row_ptr, &col_indices, n_neurons);
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    /// Number of neurons represented by this topology.
    #[inline(always)]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Borrow the validated topology in O(1), without rescanning CSR arrays.
    #[inline(always)]
    pub fn as_topology(&self) -> ValidatedCsrTopology<'_> {
        ValidatedCsrTopology::from_validated(&self.row_ptr, &self.col_indices, self.n_neurons)
    }

    /// Return the owned row pointer snapshot.
    #[inline(always)]
    pub fn row_ptr(&self) -> &[i32] {
        &self.row_ptr
    }

    /// Return the owned column-index snapshot.
    #[inline(always)]
    pub fn col_indices(&self) -> &[i32] {
        &self.col_indices
    }
}
