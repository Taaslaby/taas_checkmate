//! Version migration for .taas format.
//!
//! # Migration chain
//!
//! | From | To | Changes |
//! |------|----|---------|
//! | 0.1.0 | 0.2.0 | TriOscillator: 27 per-field Vec → grouped [27,n] tensor + scalars |
//!
//! Old files are auto-migrated on read (in `TaasReader::from_bytes` / `from_file`).
//! The migration uses the high-level reader/writer API for correctness.

use crate::types::TAAS_VERSION;
use crate::reader::TaasReader;
use crate::writer::TaasWriter;
use crate::components::{save_trioscilator, load_trioscilator};

// ─── Public API ─────────────────────────────────────────

/// Migrate raw .taas bytes to the latest format version.
///
/// Returns migrated bytes. If the file is already at the latest version,
/// returns the original bytes unchanged.
pub fn migrate_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    // Quick parse: extract version from JSON header without full validation
    let version = TaasReader::extract_version(bytes)?;
    if version == TAAS_VERSION {
        return Ok(bytes.to_vec());
    }
    // Chain migrations in order
    match version.as_str() {
        "0.1.0" => {
            let v2 = migrate_v0_1_to_v0_2(bytes)?;
            Ok(v2)
        }
        other => Err(format!("Unknown version '{other}', cannot migrate")),
    }
}

/// Check if migration is needed for the given version string.
pub fn needs_migration(version: &str) -> bool {
    version != TAAS_VERSION
}

// ─── v0.1.0 → v0.2.0 ───────────────────────────────────

/// Migrate from v0.1.0 to v0.2.0:
/// - TriOscillator: 27 individual Vec<f32> fields → 1 grouped tensor [27, n] + scalars
/// - All other components copied as-is
/// - New CRC32 fields added by writer automatically
fn migrate_v0_1_to_v0_2(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let reader = TaasReader::from_bytes(bytes)?;
    let mut writer = TaasWriter::new();
    writer.add_metadata("migrated_from", "0.1.0");

    // Copy all metadata except ones we set
    for (k, v) in reader.metadata() {
        if k != "migrated_from" {
            writer.add_metadata(k, v);
        }
    }

    // Detect if this file has TriOscillator in old per-field format
    let has_old_tri = reader.find_component("th1").is_ok();
    let tri_migration_ok = has_old_tri
        && reader.get_f32("th1").map(|d| d.len()).unwrap_or(0) > 0
        && reader.get_f32("th1").unwrap()[0].is_finite(); // quick sanity check

    // Try to migrate TriOscillator to grouped format
    let skip_tri_components = if tri_migration_ok {
        let n = reader.get_f32("th1").unwrap().len();
        match load_trioscilator(&reader, n) {
            Ok(tri) => {
                save_trioscilator(&tri, &mut writer);
                true // success — skip old fields in copy loop
            }
            Err(_) => {
                false // fallback: copy old per-field components as-is
            }
        }
    } else {
        false
    };

    // Copy all other components (skip old TriOscillator fields if migrated)
    for comp in reader.list_components() {
        let name = &comp.name;
        let ctype = &comp.component_type;

        // Skip old TriOscillator per-field components (save_trioscilator wrote them all)
        if skip_tri_components && ctype == "TriOscillator" && name != "tri_state" {
            continue;
        }

        // Copy all other components unchanged
        match comp.dtype.as_str() {
            "f32" => {
                if let Ok(data) = reader.get_f32(name) {
                    if comp.shape.len() > 1 {
                        writer.add_f32_shape(name, ctype, &data, comp.shape.clone());
                    } else {
                        writer.add_f32(name, ctype, &data);
                    }
                }
            }
            "i32" => {
                if let Ok(data) = reader.get_i32(name) {
                    if comp.shape.len() > 1 {
                        writer.add_i32_shape(name, ctype, &data, comp.shape.clone());
                    } else {
                        writer.add_i32(name, ctype, &data);
                    }
                }
            }
            "i64" => {
                if let Ok(data) = reader.get_i64(name) {
                    writer.add_i64(name, ctype, &data);
                }
            }
            "u64" => {
                if let Ok(data) = reader.get_u64(name) {
                    writer.add_u64(name, ctype, &data);
                }
            }
            "u8" => {
                if let Ok(data) = reader.get_u8(name) {
                    writer.add_u8(name, ctype, &data);
                }
            }
            _ => {}
        }
    }

    writer.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use taas_t3::{Avx2Backend, TriOscillator, NlmsDecoder};

    /// Test that needs_migration correctly identifies old versions.
    #[test]
    fn test_needs_migration() {
        assert!(needs_migration("0.1.0"));
        assert!(!needs_migration(TAAS_VERSION));
        assert!(needs_migration("0.0.9"));
    }

    /// Test migration of simple v0.1.0 file (no TriOscillator).
    /// Old format without CRC32 → gets CRC32 added automatically.
    #[test]
    fn test_migrate_simple_v0_1() {
        // Create a v0.1.0-format file: write raw bytes without CRC32
        // We'll create it via TaasWriter and then strip the CRC32 fields
        // to simulate old format
        let mut writer = TaasWriter::new();
        writer.add_f32("test", "Test", &[1.0, 2.0, 3.0]);
        writer.add_metadata("model", "test");

        // Force version to 0.1.0 by re-writing the manifest
        let mut raw = writer.to_bytes().unwrap();
        // Replace version string in the JSON header
        let old_v = b"0.2.0";
        let new_v = b"0.1.0";
        if let Some(pos) = raw.windows(5).position(|w| w == old_v) {
            raw[pos..pos+5].copy_from_slice(new_v);
        }

        // Migrate
        let migrated = migrate_bytes(&raw).unwrap();
        let reader = TaasReader::from_bytes(&migrated).unwrap();

        assert_eq!(reader.manifest.version, TAAS_VERSION);
        assert_eq!(reader.get_f32("test").unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(reader.metadata().get("model").unwrap(), "test");
        assert_eq!(reader.metadata().get("migrated_from").unwrap(), "0.1.0");
        // Should have CRC32 now
        assert!(reader.find_component("test").unwrap().crc32.is_some());
    }

    /// Test migration of TriOscillator v0.1.0 (per-field) → v0.2.0 (grouped).
    #[test]
    fn test_migrate_tri_oscillator_v0_1() {
        // Create a v0.1.0-style TriOscillator using the old per-field save format
        let n = 16;
        let tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

        // Write using the OLD format: save each field individually
        // (simulate v0.1.0 writer by calling writer.add_f32 for each field)
        let mut writer = TaasWriter::new();
        writer.add_metadata("type", "TriOscillator");
        writer.add_metadata("n", &n.to_string());

        // Per-field save (old format)
        writer.add_f32("th1", "TriOscillator", &tri.th1);
        writer.add_f32("th2", "TriOscillator", &tri.th2);
        writer.add_f32("th3", "TriOscillator", &tri.th3);
        writer.add_f32("prev_th1", "TriOscillator", &tri.prev_th1);
        writer.add_f32("prev_th2", "TriOscillator", &tri.prev_th2);
        writer.add_f32("prev_th3", "TriOscillator", &tri.prev_th3);
        writer.add_f32("r1", "TriOscillator", &tri.r1);
        writer.add_f32("r2", "TriOscillator", &tri.r2);
        writer.add_f32("r3", "TriOscillator", &tri.r3);
        writer.add_f32("prev_r1", "TriOscillator", &tri.prev_r1);
        writer.add_f32("prev_r2", "TriOscillator", &tri.prev_r2);
        writer.add_f32("prev_r3", "TriOscillator", &tri.prev_r3);
        writer.add_f32("omega1", "TriOscillator", &tri.omega1);
        writer.add_f32("omega2", "TriOscillator", &tri.omega2);
        writer.add_f32("omega3", "TriOscillator", &tri.omega3);
        writer.add_f32("drive1", "TriOscillator", &tri.drive1);
        writer.add_f32("drive2", "TriOscillator", &tri.drive2);
        writer.add_f32("drive3", "TriOscillator", &tri.drive3);
        writer.add_f32("sin1", "TriOscillator", &tri.sin1);
        writer.add_f32("cos1", "TriOscillator", &tri.cos1);
        writer.add_f32("sin2", "TriOscillator", &tri.sin2);
        writer.add_f32("cos2", "TriOscillator", &tri.cos2);
        writer.add_f32("sin3", "TriOscillator", &tri.sin3);
        writer.add_f32("cos3", "TriOscillator", &tri.cos3);
        writer.add_f32("coupling1", "TriOscillator", &tri.coupling1);
        writer.add_f32("coupling2", "TriOscillator", &tri.coupling2);
        writer.add_f32("coupling3", "TriOscillator", &tri.coupling3);
        writer.add_scalar_f32("w12", "TriOscillator", tri.w12);
        writer.add_scalar_f32("w13", "TriOscillator", tri.w13);
        writer.add_scalar_f32("w23", "TriOscillator", tri.w23);
        writer.add_scalar_f32("alpha", "TriOscillator", tri.alpha);
        writer.add_scalar_f32("drive_gain", "TriOscillator", tri.drive_gain);
        writer.add_scalar_f32("epsilon", "TriOscillator", tri.epsilon);
        writer.add_scalar_f32("phase_gate_gain", "TriOscillator", tri.phase_gate_gain);
        writer.add_scalar_f32("gamma1", "TriOscillator", tri.gamma1);
        writer.add_scalar_f32("gamma2", "TriOscillator", tri.gamma2);
        writer.add_scalar_f32("gamma3", "TriOscillator", tri.gamma3);
        writer.add_scalar_bool("initialised", "TriOscillator", tri.initialised);

        // Force version to 0.1.0
        let mut raw = writer.to_bytes().unwrap();
        let old_v = b"0.2.0";
        let new_v = b"0.1.0";
        if let Some(pos) = raw.windows(5).position(|w| w == old_v) {
            raw[pos..pos+5].copy_from_slice(new_v);
        }

        // Migrate
        let migrated = migrate_bytes(&raw).unwrap();
        let reader = TaasReader::from_bytes(&migrated).unwrap();

        assert_eq!(reader.manifest.version, TAAS_VERSION);

        // Check that old per-field components are gone
        assert!(reader.find_component("th1").is_err());
        assert!(reader.find_component("th2").is_err());

        // Check that new grouped tensor exists
        let tri_state = reader.find_component("tri_state").unwrap();
        assert_eq!(tri_state.shape, vec![27, n]);

        // Check that scalars are preserved
        // Note: TriOscillator::new(n, freq_min, freq_max, backend) — NOT (n, alpha, drive_gain, backend)
        // alpha defaults to 1.0, drive_gain defaults to 10.0
        assert!((reader.get_scalar_f32("alpha").unwrap() - 1.0).abs() < 1e-6);
        assert!((reader.get_scalar_f32("drive_gain").unwrap() - 10.0).abs() < 1e-6);

        // Load via load_trioscilator and verify
        let loaded = load_trioscilator(&reader, n).unwrap();
        assert_eq!(loaded.th1, tri.th1);
        assert_eq!(loaded.th2, tri.th2);
        assert_eq!(loaded.alpha, tri.alpha);
        assert_eq!(loaded.initialised, tri.initialised);
    }

    /// Test that migration doesn't change already-current format.
    #[test]
    fn test_migrate_already_current() {
        let mut writer = TaasWriter::new();
        writer.add_f32("test", "Test", &[1.0, 2.0, 3.0]);
        let bytes = writer.to_bytes().unwrap();

        // Migrate should return same bytes
        let migrated = migrate_bytes(&bytes).unwrap();
        assert_eq!(migrated, bytes);
    }

    /// Test migration of mixed file: TriOscillator + NlmsDecoder.
    #[test]
    fn test_migrate_tri_and_nlms() {
        let n = 16;
        let tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        let dec = NlmsDecoder::new(n, 0.01, 1e-8);

        // Write old-format TriOscillator + NlmsDecoder
        let mut writer = TaasWriter::new();
        // Only the essential TriOscillator fields for a quick test
        writer.add_f32("th1", "TriOscillator", &tri.th1);
        writer.add_f32("th2", "TriOscillator", &tri.th2);
        writer.add_scalar_f32("alpha", "TriOscillator", tri.alpha);
        writer.add_scalar_bool("initialised", "TriOscillator", tri.initialised);
        // NlmsDecoder (unchanged format)
        writer.add_f32("nlms_weights", "NlmsDecoder", &dec.weights);
        writer.add_scalar_f32("nlms_lr", "NlmsDecoder", dec.lr);

        let mut raw = writer.to_bytes().unwrap();
        let old_v = b"0.2.0";
        let new_v = b"0.1.0";
        if let Some(pos) = raw.windows(5).position(|w| w == old_v) {
            raw[pos..pos+5].copy_from_slice(new_v);
        }

        let migrated = migrate_bytes(&raw).unwrap();
        let reader = TaasReader::from_bytes(&migrated).unwrap();

        // TriOscillator is now grouped
        assert!(reader.find_component("tri_state").is_ok());
        assert!(reader.find_component("th1").is_err());

        // NlmsDecoder is preserved
        assert!(reader.find_component("nlms_weights").is_ok());
        assert!(reader.find_component("nlms_lr").is_ok());
        assert!((reader.get_scalar_f32("nlms_lr").unwrap() - 0.01).abs() < 1e-6);
    }
}
