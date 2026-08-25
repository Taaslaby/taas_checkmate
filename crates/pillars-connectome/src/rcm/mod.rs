//! RCM (Reverse Cuthill-McKee) & Spectral Graph Reordering.
//!
//! Reorders connectome neurons to minimize CSR matrix bandwidth and CPU cache misses.

pub mod bfs;
pub mod permute;
pub mod spectral;

pub use bfs::*;
pub use permute::*;
pub use spectral::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ConnectomeData;

    #[test]
    fn test_pseudo_peripheral() {
        let n = 6;
        let row_ptr = vec![0i32, 2, 3, 4, 5, 6, 7];
        let col_indices = vec![1, 5, 2, 3, 4, 0, 0];
        let n_syn = col_indices.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values: vec![1.0; n_syn],
            nt_types: vec![1.0; n],
            receptor_types: vec![],
            groups: vec![],
            phenotype: crate::data::NtPhenotype::default(),
        };
        let adj = build_adjacency(&dat);
        let root = pseudo_peripheral_node(&adj);
        println!("pseudo_peripheral root = {root}");
        assert!(root < n, "root {root} out of bounds");
    }

    #[test]
    fn test_rcm_ordering() {
        let n = 6;
        let row_ptr = vec![0i32, 2, 3, 4, 5, 6, 7];
        let col_indices = vec![1, 5, 2, 3, 4, 0, 0];
        let n_syn = col_indices.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values: vec![1.0; n_syn],
            nt_types: vec![1.0; n],
            receptor_types: vec![],
            groups: vec![],
            phenotype: crate::data::NtPhenotype::default(),
        };
        let adj = build_adjacency(&dat);
        let perm = reverse_cuthill_mckee(&adj);

        let mut sorted = perm.clone();
        sorted.sort();
        for (i, &p) in sorted.iter().enumerate() {
            assert_eq!(p, i, "perm is not a valid permutation at {i}");
        }
        println!("RCM perm: {:?}", perm);
    }

    #[test]
    fn test_bandwidth_reduction() {
        let n = 32;
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            let post = (i + 1) % n;
            col_indices.push(post as i32);
            values.push(1.0);
            row_ptr[i + 1] = i as i32 + 1;
        }

        let n_syn = values.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values,
            nt_types: vec![1.0; n],
            receptor_types: vec![],
            groups: vec![],
            phenotype: crate::data::NtPhenotype::default(),
        };

        let bw_before = bandwidth(&dat.row_ptr, &dat.col_indices);
        let adj = build_adjacency(&dat);
        let perm = reverse_cuthill_mckee(&adj);
        let (new_rp, new_ci, _, _) = apply_permutation_to_csr(
            &dat.row_ptr,
            &dat.col_indices,
            &dat.values,
            &dat.nt_types,
            &perm,
        );
        let bw_after = bandwidth(&new_rp, &new_ci);
        println!("Bandwidth BEFORE: {bw_before}");
        println!("Bandwidth AFTER:  {bw_after}");

        assert!(bw_after <= bw_before, "bandwidth should not increase");
    }

    #[test]
    fn test_fiedler_vector_simple() {
        let adj = vec![
            vec![1],    // 0
            vec![0, 2], // 1
            vec![1, 3], // 2
            vec![2],    // 3
        ];
        let fv = fiedler_vector(&adj, 100, 1e-4);
        assert_eq!(fv.len(), 4);
        for i in 0..3 {
            assert!(
                fv[i] != fv[i + 1],
                "Fiedler vector should distinguish adjacent nodes"
            );
        }
        println!("Fiedler vector (path of 4): {:?}", fv);
    }
}
