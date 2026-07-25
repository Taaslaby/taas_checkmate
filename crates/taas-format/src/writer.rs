use std::collections::HashMap;
use std::io::{Write, Cursor};
use std::path::Path;
use crate::types::*;

pub struct TaasWriter {
    components: Vec<ComponentEntry>,
    tensor_data: Vec<u8>,
    metadata: HashMap<String, String>,
    compression: Option<i32>,  // zstd compression level if Some
}

/// Helper: compute CRC32 for a byte slice.
fn compute_crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

impl TaasWriter {
    pub fn new() -> Self {
        Self { components: Vec::new(), tensor_data: Vec::new(), metadata: HashMap::new(), compression: None }
    }

    /// Enable zstd compression for all subsequent components.
    /// `level`: 1 (fast) to 22 (max), default 3.
    pub fn set_compression(&mut self, level: i32) {
        self.compression = Some(level.clamp(1, 22));
    }

    /// Disable compression for subsequent components.
    pub fn disable_compression(&mut self) {
        self.compression = None;
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    fn push_component(&mut self, name: &str, component_type: &str, dtype: &str, shape: Vec<usize>, data: &[u8]) {
        let crc32 = Some(compute_crc32(data));

        // Optional zstd compression
        let (stored, compression, uncompressed_size) = match self.compression {
            Some(level) => {
                match zstd::bulk::compress(data, level) {
                    Ok(c) => (c, Some("zstd".to_string()), Some(data.len() as u64)),
                    Err(_) => (data.to_vec(), None, None),  // fallback: store uncompressed
                }
            }
            None => (data.to_vec(), None, None),
        };

        let offset = self.tensor_data.len() as u64;
        let size = stored.len() as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: dtype.to_string(),
            shape,
            offset,
            size,
            crc32,
            compression,
            uncompressed_size,
        });
        self.tensor_data.extend_from_slice(&stored);
    }

    pub fn add_f32(&mut self, name: &str, component_type: &str, data: &[f32]) {
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "f32", vec![data.len()], bytes);
    }

    pub fn add_f32_shape(&mut self, name: &str, component_type: &str, data: &[f32], shape: Vec<usize>) {
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "f32", shape, bytes);
    }

    pub fn add_i32(&mut self, name: &str, component_type: &str, data: &[i32]) {
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "i32", vec![data.len()], bytes);
    }

    pub fn add_i32_shape(&mut self, name: &str, component_type: &str, data: &[i32], shape: Vec<usize>) {
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "i32", shape, bytes);
    }

    pub fn add_i64(&mut self, name: &str, component_type: &str, data: &[i64]) {
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "i64", vec![data.len()], bytes);
    }

    pub fn add_u64(&mut self, name: &str, component_type: &str, data: &[u64]) {
        let bytes = bytemuck::cast_slice(data);
        self.push_component(name, component_type, "u64", vec![data.len()], bytes);
    }

    pub fn add_u8(&mut self, name: &str, component_type: &str, data: &[u8]) {
        self.push_component(name, component_type, "u8", vec![data.len()], data);
    }

    pub fn add_scalar_f32(&mut self, name: &str, component_type: &str, value: f32) {
        self.add_f32(name, component_type, &[value]);
    }

    pub fn add_scalar_bool(&mut self, name: &str, component_type: &str, value: bool) {
        self.add_i32(name, component_type, &[if value { 1 } else { 0 }]);
    }

    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), String> {
        let manifest = TaasManifest {
            version: TAAS_VERSION.to_string(),
            created: crate::chrono_or_fallback(),
            components: self.components.clone(),
            metadata: self.metadata.clone(),
        };
        let header_json = serde_json::to_string(&manifest).map_err(|e| format!("Failed to serialize manifest: {e}"))?;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        writer.write_all(TAAS_MAGIC).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(&header_len.to_le_bytes()).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(header_bytes).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(&self.tensor_data).map_err(|e| format!("Write error: {e}"))?;
        Ok(())
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let mut file = std::fs::File::create(path.as_ref()).map_err(|e| format!("Cannot create file: {e}"))?;
        self.save(&mut file)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        self.save(&mut Cursor::new(&mut buf))?;
        Ok(buf)
    }
}

impl Default for TaasWriter {
    fn default() -> Self { Self::new() }
}
