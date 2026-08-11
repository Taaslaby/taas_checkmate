#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::should_implement_trait)]

//! # `.taas` format — универсальный снапшот для SNN моделей и осцилляторов.
//!
//! ## Бинарный формат
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │  MAGIC: b"TAAS1" (5 bytes)           │
//! │  HEADER_LEN: u64 LE (8 bytes)        │
//! │  HEADER: UTF-8 JSON (HEADER_LEN)     │
//! │  ├── version: "0.1.0"               │
//! │  ├── created: "2026-06-19T..."      │
//! │  ├── components: [ ... ]            │
//! │  └── metadata: { ... }              │
//! │  TENSOR_DATA: [...]                  │
//! └──────────────────────────────────────┘
//! ```
//!
//! Каждый компонент в `components` массиве:
//! ```json
//! {"name":"tri_osc", "type":"TriOscillator",
//!  "dtype":"f32", "shape":[128, 3],
//!  "offset":0, "size":1536}
//! ```

pub mod components;
pub mod inspect;
pub mod migration;
pub mod reader;
pub mod types;
pub mod writer;

pub use components::*;
pub use inspect::inspect_taas;
pub use migration::migrate_bytes;
pub use reader::TaasReader;
pub use types::{ComponentEntry, TaasManifest, TensorDtype, TAAS_MAGIC, TAAS_VERSION};
pub use writer::TaasWriter;

// ─── Fallback timestamp (without chrono dependency) ─────

pub(crate) fn chrono_or_fallback() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs_rem = time_secs % 60;
    let year = 1970 + (days as f64 / 365.25) as u64;
    format!("{year}-01-01T{hours:02}:{mins:02}:{secs_rem:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use taas_t3::{Avx2Backend, KalmanDecoder, NlmsDecoder, TriOscillator};

    #[test]
    fn test_roundtrip_taas_basic() {
        let mut writer = TaasWriter::new();
        writer.add_f32("test_tensor", "Test", &[1.0, 2.0, 3.0, 4.0]);
        writer.add_i32("test_int", "Test", &[10, 20, 30]);
        writer.add_scalar_f32("test_scalar", "Test", 42.0);
        writer.add_scalar_bool("test_bool", "Test", true);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        assert_eq!(
            reader.get_f32("test_tensor").unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(reader.get_i32("test_int").unwrap(), &[10, 20, 30]);
        assert_eq!(reader.get_scalar_f32("test_scalar").unwrap(), 42.0);
        assert!(reader.get_scalar_bool("test_bool").unwrap());
    }

    #[test]
    fn test_validate_integrity() {
        let mut writer = TaasWriter::new();
        writer.add_f32("data", "Test", &[1.0; 100]);
        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        assert!(reader.validate().is_ok());
    }

    #[test]
    fn test_invalid_magic() {
        let bad = b"NOTTAAS\x00\x00\x00\x00\x00\x00\x00\x00{}\x00";
        assert!(TaasReader::from_bytes(bad).is_err());
    }

    #[test]
    fn test_tri_oscillator_roundtrip() {
        use taas_t3::TriOscillator;
        let tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        let mut writer = TaasWriter::new();
        save_trioscilator(&tri, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_trioscilator(&reader, 16).unwrap();
        assert_eq!(loaded.n, tri.n);
        assert_eq!(loaded.th1, tri.th1);
        assert_eq!(loaded.alpha, tri.alpha);
        assert_eq!(loaded.initialised, tri.initialised);
    }

    #[test]
    fn test_nlms_decoder_roundtrip() {
        let mut dec = NlmsDecoder::new(8, 0.01, 1e-8);
        dec.weights = vec![0.5, -0.3, 0.8, 0.1, -0.5, 0.3, 0.2, -0.1];
        dec.bias[0] = 0.25;

        let mut writer = TaasWriter::new();
        save_nlms_decoder(&dec, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_nlms_decoder(&reader, 8).unwrap();
        assert_eq!(loaded.weights, dec.weights);
        assert!((loaded.bias[0] - 0.25).abs() < 1e-6);
        assert!((loaded.lr - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_kalman_decoder_roundtrip() {
        let mut dec = KalmanDecoder::new(8, 0.001, 0.1);
        dec.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        dec.bias = 0.5;

        let mut writer = TaasWriter::new();
        save_kalman_decoder(&dec, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_kalman_decoder(&reader, 8).unwrap();
        assert_eq!(loaded.weights, dec.weights);
        assert!((loaded.bias - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_population_roundtrip() {
        use pillars::zoo::LIFModel;
        use pillars::zoo::Model;
        use pillars::NeuralPopulation;
        let mut pop = NeuralPopulation::<LIFModel>::new(13);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);
        pop.fill_state(1, 0.0);
        for i in 0..13 {
            pop.state_slice_mut(0)[i] = -70.0 + i as f32;
        }

        let mut writer = TaasWriter::new();
        save_population(&pop, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_population::<LIFModel>(&reader).unwrap();
        assert_eq!(loaded.len(), pop.len());
        for c in 0..LIFModel::N_PARAMS {
            assert_eq!(loaded.param_slice(c), pop.param_slice(c));
        }
        for c in 0..LIFModel::N_STATE {
            assert_eq!(loaded.state_slice(c), pop.state_slice(c));
        }
    }

    #[test]
    fn test_sparse_graph_roundtrip() {
        use pillars::api::SparseGraph;
        let g = SparseGraph::from_triplets(
            4,
            &[0, 1, 2, 3, 0],
            &[1, 2, 3, 0, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1, 2, 3, 4, 5],
        );

        let mut writer = TaasWriter::new();
        save_sparse_graph(&g, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_sparse_graph(&reader).unwrap();
        assert_eq!(loaded.num_neurons(), g.num_neurons());
        assert_eq!(loaded.num_synapses(), g.num_synapses());
        assert_eq!(loaded.row_ptr(), g.row_ptr());
        assert_eq!(loaded.col_idx(), g.col_idx());
        assert_eq!(loaded.delays(), g.delays());
        assert_eq!(loaded.weights(), g.weights());
    }

    #[test]
    fn test_multiple_components() {
        let tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        let dec = NlmsDecoder::new(16, 0.01, 1e-8);

        let mut writer = TaasWriter::new();
        writer.add_metadata("description", "Test multi-component");
        save_trioscilator(&tri, &mut writer);
        save_nlms_decoder(&dec, &mut writer);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();

        // TriOscillator: 1 grouped tensor + 10 scalars + 1 bool = 12 components
        // NlmsDecoder: 2 tensors + 2 scalars = 4 components
        assert_eq!(reader.manifest.components.len(), 12 + 4);
        assert_eq!(
            reader.metadata().get("description").unwrap(),
            "Test multi-component"
        );
        assert!(reader.find_component("tri_state").is_ok());
        assert!(reader.find_component("nlms_weights").is_ok());
    }

    #[test]
    fn test_inspect_output() {
        let mut writer = TaasWriter::new();
        writer.add_f32("test", "Test", &[1.0, 2.0, 3.0]);
        writer.add_metadata("model", "izhikevich");
        let bytes = writer.to_bytes().unwrap();
        let tmp = "test_inspect.taas";
        let mut f = std::fs::File::create(tmp).unwrap();
        use std::io::Write;
        f.write_all(&bytes).unwrap();
        drop(f);

        let output = inspect_taas(tmp).unwrap();
        assert!(output.contains("test"));
        assert!(output.contains("izhikevich"));
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_load_tri_oscillator() {
        let tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        let mut writer = TaasWriter::new();
        save_trioscilator(&tri, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_trioscilator(&reader, 16).unwrap();
        assert_eq!(loaded.n, tri.n);
        assert_eq!(loaded.initialised, tri.initialised);
    }

    #[test]
    fn test_connectome_roundtrip() {
        use pillars_connectome::ConnectomeData;
        let conn = ConnectomeData::new(
            vec![0i32, 2, 5, 8],
            vec![0i32, 1, 0, 2, 1, 2, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1.0, -1.0, 1.0],
            vec![("mn", vec![0, 1]), ("dn", vec![2])],
        );

        let mut writer = TaasWriter::new();
        save_connectome(&conn, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_connectome(&reader).unwrap();

        assert_eq!(loaded.n_neurons, conn.n_neurons);
        assert_eq!(loaded.n_synapses, conn.n_synapses);
        assert_eq!(loaded.row_ptr, conn.row_ptr);
        assert_eq!(loaded.col_indices, conn.col_indices);
        assert_eq!(loaded.values, conn.values);
        assert_eq!(loaded.nt_types, conn.nt_types);
        assert_eq!(loaded.groups.len(), conn.groups.len());
        assert_eq!(loaded.groups[0].name, "mn");
        assert_eq!(loaded.groups[1].indices, vec![2usize]);
    }

    #[test]
    fn test_zstd_compression_roundtrip() {
        let mut writer = TaasWriter::new();
        writer.set_compression(3); // zstd level 3
        writer.add_f32("data", "Test", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        writer.add_i32("int_data", "Test", &[10, 20, 30, 40]);
        writer.add_scalar_f32("scalar", "Test", 42.0);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();

        // Verify compression header fields
        let data_entry = reader.find_component("data").unwrap();
        assert_eq!(data_entry.compression.as_deref(), Some("zstd"));
        let uncomp = data_entry.uncompressed_size.unwrap();
        assert!(uncomp >= 5 * 4, "uncompressed_size={uncomp} < 20");
        // zstd may not reduce size for tiny data (< ~100 bytes) due to framing overhead
        // but the fields must be set correctly

        // Verify data round-trip
        assert_eq!(reader.get_f32("data").unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(reader.get_i32("int_data").unwrap(), &[10, 20, 30, 40]);
        assert_eq!(reader.get_scalar_f32("scalar").unwrap(), 42.0);
        assert!(reader.validate().is_ok());

        // Test mixed: compress some components, not others
        let mut writer2 = TaasWriter::new();
        writer2.add_f32("raw", "Test", &[1.0; 16]); // uncompressed
        writer2.set_compression(3);
        writer2.add_f32("compressed", "Test", &[2.0; 16]);
        writer2.disable_compression();
        writer2.add_f32("raw2", "Test", &[3.0; 16]); // uncompressed again

        let bytes2 = writer2.to_bytes().unwrap();
        let reader2 = TaasReader::from_bytes(&bytes2).unwrap();
        assert!(reader2.find_component("raw").unwrap().compression.is_none());
        assert_eq!(
            reader2
                .find_component("compressed")
                .unwrap()
                .compression
                .as_deref(),
            Some("zstd")
        );
        assert!(reader2
            .find_component("raw2")
            .unwrap()
            .compression
            .is_none());
        assert_eq!(reader2.get_f32("raw").unwrap(), &[1.0; 16]);
        assert_eq!(reader2.get_f32("compressed").unwrap(), &[2.0; 16]);
        assert_eq!(reader2.get_f32("raw2").unwrap(), &[3.0; 16]);
    }
}
