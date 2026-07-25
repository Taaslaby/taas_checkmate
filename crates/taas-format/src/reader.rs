use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use crate::types::*;

enum DataSource {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl DataSource {
    fn as_slice(&self) -> &[u8] {
        match self {
            DataSource::Owned(v) => v.as_slice(),
            DataSource::Mapped(m) => m.as_ref(),
        }
    }
}

pub struct TaasReader {
    pub manifest: TaasManifest,
    source: DataSource,
    data_offset: usize,
}

impl TaasReader {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (manifest, data_offset) = Self::parse_header(bytes)?;
        Ok(Self {
            manifest,
            source: DataSource::Owned(bytes.to_vec()),
            data_offset,
        })
    }

    /// Extract version from raw TAAS bytes (minimal parse, for migration check).
    pub(crate) fn extract_version(bytes: &[u8]) -> Result<String, String> {
        if bytes.len() < 13 { return Err("File too short".to_string()); }
        if &bytes[0..5] != crate::types::TAAS_MAGIC {
            return Err("Invalid magic".to_string());
        }
        let header_len = u64::from_le_bytes(
            bytes[5..13].try_into().map_err(|_| "Bad header length".to_string())?
        ) as usize;
        let header_start = 13;
        if header_start + header_len > bytes.len() {
            return Err("Header truncated".to_string());
        }
        let header_str = std::str::from_utf8(&bytes[header_start..header_start + header_len])
            .map_err(|e| format!("Invalid UTF-8: {e}"))?;
        if let Some(pos) = header_str.find(r#""version""#) {
            let after = &header_str[pos + 9..];
            let start = after.find('"').ok_or("Bad version format")? + 1;
            let end = after[start..].find('"').ok_or("Bad version format")?;
            Ok(after[start..start + end].to_string())
        } else {
            Err("Version field not found".to_string())
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("Cannot open file: {e}"))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("Cannot mmap file: {e}"))?
        };
        let bytes = mmap.as_ref();
        let (manifest, data_offset) = Self::parse_header(bytes)?;
        Self::validate_offsets(&manifest, bytes, data_offset)?;
        Ok(Self {
            manifest,
            source: DataSource::Mapped(mmap),
            data_offset,
        })
    }

    fn parse_header(bytes: &[u8]) -> Result<(TaasManifest, usize), String> {
        let mut pos = 0usize;
        if bytes.len() < 5 + 8 {
            return Err("File too short".to_string());
        }
        if &bytes[0..5] != TAAS_MAGIC {
            return Err(format!("Invalid magic: expected TAAS1, got {:?}", &bytes[0..5]));
        }
        pos += 5;
        let header_len = u64::from_le_bytes(
            bytes[pos..pos+8].try_into().map_err(|_| "Header length field truncated".to_string())?
        ) as usize;
        pos += 8;
        if pos + header_len > bytes.len() {
            return Err("Header truncated".to_string());
        }
        let header_str = std::str::from_utf8(&bytes[pos..pos+header_len])
            .map_err(|e| format!("Invalid UTF-8 in header: {e}"))?;
        let manifest: TaasManifest = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid manifest JSON: {e}"))?;
        pos += header_len;
        Ok((manifest, pos))
    }

    fn validate_offsets(manifest: &TaasManifest, bytes: &[u8], data_offset: usize) -> Result<(), String> {
        for comp in &manifest.components {
            let start = data_offset + comp.offset as usize;
            let end = start + comp.size as usize;
            if end > bytes.len() {
                return Err(format!(
                    "Component '{}' (offset={}, size={}) exceeds data (len={})",
                    comp.name, comp.offset, comp.size, bytes.len() - data_offset
                ));
            }
        }
        Ok(())
    }

    fn data_bytes(&self) -> &[u8] {
        &self.source.as_slice()[self.data_offset..]
    }

    fn read_bytes(&self, entry: &ComponentEntry) -> Result<Vec<u8>, String> {
        let data = self.data_bytes();
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > data.len() {
            return Err(format!("Component '{}' data out of bounds", entry.name));
        }
        let slice = &data[start..end];

        // Decompress if needed
        let decompressed = match entry.compression.as_deref() {
            Some("zstd") => {
                let mut decoder = zstd::Decoder::new(slice)
                    .map_err(|e| format!("Component '{}' zstd decompression failed: {e}", entry.name))?;
                let mut buf = Vec::with_capacity(entry.uncompressed_size.unwrap_or(0) as usize);
                decoder.read_to_end(&mut buf)
                    .map_err(|e| format!("Component '{}' zstd decompression failed: {e}", entry.name))?;
                buf
            }
            _ => slice.to_vec(),
        };

        // Verify CRC32 on (decompressed) data
        if let Some(expected_crc) = entry.crc32 {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&decompressed);
            let actual_crc = hasher.finalize();
            if actual_crc != expected_crc {
                return Err(format!(
                    "Component '{}' CRC32 mismatch: expected {:08x}, got {:08x}",
                    entry.name, expected_crc, actual_crc
                ));
            }
        }

        Ok(decompressed)
    }

    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "f32" {
            return Err(format!("Component '{name}' is not f32, got {}", entry.dtype));
        }
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
        if entry.dtype != "i32" {
            return Err(format!("Component '{name}' is not i32, got {}", entry.dtype));
        }
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
        if entry.dtype != "i64" {
            return Err(format!("Component '{name}' is not i64, got {}", entry.dtype));
        }
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
        if entry.dtype != "u64" {
            return Err(format!("Component '{name}' is not u64, got {}", entry.dtype));
        }
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

    pub fn get_u8(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "u8" {
            return Err(format!("Component '{name}' is not u8, got {}", entry.dtype));
        }
        let src = self.read_bytes(entry)?;
        Ok(src.to_vec())
    }

    pub fn get_scalar_f32(&self, name: &str) -> Result<f32, String> {
        let slice = self.get_f32(name)?;
        if slice.len() != 1 {
            return Err(format!("Component '{name}' is not scalar f32, len={}", slice.len()));
        }
        Ok(slice[0])
    }

    pub fn get_scalar_bool(&self, name: &str) -> Result<bool, String> {
        let slice = self.get_i32(name)?;
        if slice.len() != 1 {
            return Err(format!("Component '{name}' is not scalar bool, len={}", slice.len()));
        }
        Ok(slice[0] != 0)
    }

    pub fn find_component(&self, name: &str) -> Result<&ComponentEntry, String> {
        self.manifest.components.iter()
            .find(|c| c.name == name)
            .ok_or_else(|| format!("Component '{name}' not found in manifest"))
    }

    pub fn list_components(&self) -> &[ComponentEntry] { &self.manifest.components }
    pub fn metadata(&self) -> &HashMap<String, String> { &self.manifest.metadata }

    pub fn total_size(&self) -> usize { self.source.as_slice().len() }

    pub fn validate(&self) -> Result<(), String> {
        // Validate uses read_bytes which handles CRC32 + decompression
        for comp in &self.manifest.components {
            self.read_bytes(comp)?;
        }
        Ok(())
    }
}
