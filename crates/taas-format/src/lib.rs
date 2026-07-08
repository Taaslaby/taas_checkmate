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

pub mod types;
pub mod writer;
pub mod reader;
pub mod components;
pub mod inspect;

pub use types::{TAAS_MAGIC, TAAS_VERSION, TensorDtype, ComponentEntry, TaasManifest};
pub use writer::TaasWriter;
pub use reader::TaasReader;
pub use components::*;
pub use inspect::inspect_taas;

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
    use taas_t3::{Avx2Backend, TriOscillator, NlmsDecoder, KalmanDecoder};
    use taas_core::neuron::SornNetwork;

    #[test]
    fn test_roundtrip_taas_basic() {
        let mut writer = TaasWriter::new();
        writer.add_f32("test_tensor", "Test", &[1.0, 2.0, 3.0, 4.0]);
        writer.add_i32("test_int", "Test", &[10, 20, 30]);
        writer.add_scalar_f32("test_scalar", "Test", 42.0);
        writer.add_scalar_bool("test_bool", "Test", true);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        assert_eq!(reader.get_f32("test_tensor").unwrap(), &[1.0, 2.0, 3.0, 4.0]);
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
    fn test_sorn_network_roundtrip() {
        let sorn = SornNetwork::new(8, 0.1, 5.0);

        let mut writer = TaasWriter::new();
        save_sorn_network(&sorn, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_sorn_network(&reader, 8).unwrap();
        assert_eq!(loaded.weights, sorn.weights);
        assert!((loaded.w_target_sum - 5.0).abs() < 1e-6);
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

        assert_eq!(reader.manifest.components.len(), 38 + 4);
        assert_eq!(reader.metadata().get("description").unwrap(), "Test multi-component");
        assert!(reader.find_component("th1").is_ok());
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
        use taas_connectome::ConnectomeData;
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
    fn test_brain_state_roundtrip() {
        use taas_connectome::sim::{BrainState, NeuronModel};
        let state = BrainState::new(10, NeuronModel::Izhikevich);
        let mut writer = TaasWriter::new();
        save_brain_state(&state, &mut writer);
        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_brain_state(&reader).unwrap();
        assert_eq!(loaded.v, state.v);
        assert_eq!(loaded.u, state.u);
        assert_eq!(loaded.n_padded, state.n_padded);
    }

    #[test]
    fn test_izhikevich_params_roundtrip() {
        use taas_connectome::sim::IzhikevichParams;
        let params = IzhikevichParams {
            a: vec![0.02; 8], b: vec![0.2; 8], c: vec![-65.0; 8], d: vec![2.0; 8],
            dt_ms: 0.5, tau_ref: 2.0,
        };
        let mut writer = TaasWriter::new();
        save_izhikevich_params(&params, &mut writer);
        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let loaded = load_izhikevich_params(&reader).unwrap();
        assert_eq!(loaded.a, params.a);
        assert_eq!(loaded.dt_ms, params.dt_ms);
        assert_eq!(loaded.tau_ref, params.tau_ref);
    }
}
