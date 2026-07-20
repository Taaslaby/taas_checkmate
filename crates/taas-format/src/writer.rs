use std::collections::HashMap;
use std::io::{Write, Cursor};
use std::path::Path;
use crate::types::*;

pub struct TaasWriter {
    components: Vec<ComponentEntry>,
    tensor_data: Vec<u8>,
    metadata: HashMap<String, String>,
}

impl TaasWriter {
    pub fn new() -> Self {
        Self { components: Vec::new(), tensor_data: Vec::new(), metadata: HashMap::new() }
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    pub fn add_f32(&mut self, name: &str, component_type: &str, data: &[f32]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "f32".to_string(), shape: vec![data.len()], offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_f32_shape(&mut self, name: &str, component_type: &str, data: &[f32], shape: Vec<usize>) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "f32".to_string(), shape, offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_i32(&mut self, name: &str, component_type: &str, data: &[i32]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "i32".to_string(), shape: vec![data.len()], offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_i32_shape(&mut self, name: &str, component_type: &str, data: &[i32], shape: Vec<usize>) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "i32".to_string(), shape, offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_i64(&mut self, name: &str, component_type: &str, data: &[i64]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 8) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "i64".to_string(), shape: vec![data.len()], offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_u64(&mut self, name: &str, component_type: &str, data: &[u64]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 8) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "u64".to_string(), shape: vec![data.len()], offset, size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    pub fn add_u8(&mut self, name: &str, component_type: &str, data: &[u8]) {
        let offset = self.tensor_data.len() as u64;
        let size = data.len() as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(), component_type: component_type.to_string(),
            dtype: "u8".to_string(), shape: vec![data.len()], offset, size,
        });
        self.tensor_data.extend_from_slice(data);
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
