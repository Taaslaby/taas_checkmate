use std::collections::HashMap;
use std::path::Path;
use crate::types::*;

pub struct TaasReader {
    pub manifest: TaasManifest,
    data: Vec<u8>,
    data_offset: usize,
}

impl TaasReader {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;
        if bytes.len() < 5 + 8 { return Err("File too short".to_string()); }
        if &bytes[0..5] != TAAS_MAGIC { return Err(format!("Invalid magic: expected TAAS1, got {:?}", &bytes[0..5])); }
        pos += 5;
        let header_len = u64::from_le_bytes(
            bytes[pos..pos+8].try_into().map_err(|_| "Header length field truncated".to_string())?
        ) as usize;
        pos += 8;
        if pos + header_len > bytes.len() { return Err("Header truncated".to_string()); }
        let header_str = std::str::from_utf8(&bytes[pos..pos+header_len])
            .map_err(|e| format!("Invalid UTF-8 in header: {e}"))?;
        let manifest: TaasManifest = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid manifest JSON: {e}"))?;
        pos += header_len;
        Ok(Self { manifest, data: bytes.to_vec(), data_offset: pos })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| format!("Cannot read file: {e}"))?;
        Self::from_bytes(&bytes)
    }

    fn read_bytes(&self, entry: &ComponentEntry) -> Result<&[u8], String> {
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() { return Err(format!("Component '{}' data out of bounds", entry.name)); }
        Ok(&self.data[start..end])
    }

    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "f32" { return Err(format!("Component '{name}' is not f32, got {}", entry.dtype)); }
        let src = self.read_bytes(entry)?;
        let n = src.len() / 4;
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            let off = i * 4;
            result[i] = f32::from_le_bytes([src[off], src[off+1], src[off+2], src[off+3]]);
        }
        Ok(result)
    }

    pub fn get_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "i32" { return Err(format!("Component '{name}' is not i32, got {}", entry.dtype)); }
        let src = self.read_bytes(entry)?;
        let n = src.len() / 4;
        let mut result = vec![0i32; n];
        for i in 0..n {
            let off = i * 4;
            result[i] = i32::from_le_bytes([src[off], src[off+1], src[off+2], src[off+3]]);
        }
        Ok(result)
    }

    pub fn get_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "i64" { return Err(format!("Component '{name}' is not i64, got {}", entry.dtype)); }
        let src = self.read_bytes(entry)?;
        let n = src.len() / 8;
        let mut result = vec![0i64; n];
        for i in 0..n {
            let off = i * 8;
            result[i] = i64::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
                src[off+4], src[off+5], src[off+6], src[off+7],
            ]);
        }
        Ok(result)
    }

    pub fn get_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "u64" { return Err(format!("Component '{name}' is not u64, got {}", entry.dtype)); }
        let src = self.read_bytes(entry)?;
        let n = src.len() / 8;
        let mut result = vec![0u64; n];
        for i in 0..n {
            let off = i * 8;
            result[i] = u64::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
                src[off+4], src[off+5], src[off+6], src[off+7],
            ]);
        }
        Ok(result)
    }

    pub fn get_scalar_f32(&self, name: &str) -> Result<f32, String> {
        let slice = self.get_f32(name)?;
        if slice.len() != 1 { return Err(format!("Component '{name}' is not scalar f32, len={}", slice.len())); }
        Ok(slice[0])
    }

    pub fn get_scalar_bool(&self, name: &str) -> Result<bool, String> {
        let slice = self.get_i32(name)?;
        if slice.len() != 1 { return Err(format!("Component '{name}' is not scalar bool, len={}", slice.len())); }
        Ok(slice[0] != 0)
    }

    pub fn find_component(&self, name: &str) -> Result<&ComponentEntry, String> {
        self.manifest.components.iter()
            .find(|c| c.name == name)
            .ok_or_else(|| format!("Component '{name}' not found in manifest"))
    }

    pub fn list_components(&self) -> &[ComponentEntry] { &self.manifest.components }
    pub fn metadata(&self) -> &HashMap<String, String> { &self.manifest.metadata }

    pub fn total_size(&self) -> usize { self.data.len() }

    pub fn validate(&self) -> Result<(), String> {
        for comp in &self.manifest.components {
            let start = comp.offset as usize;
            let end = start + comp.size as usize;
            if end > self.data.len() - self.data_offset {
                return Err(format!("Component '{}' (offset={}, size={}) exceeds data (len={})",
                    comp.name, comp.offset, comp.size, self.data.len() - self.data_offset));
            }
            if let Some(dt) = TensorDtype::from_str(&comp.dtype) {
                if start % dt.elem_size() != 0 {
                    return Err(format!("Component '{}' misaligned: offset {} not multiple of {}",
                        comp.name, start, dt.elem_size()));
                }
            }
        }
        Ok(())
    }
}
