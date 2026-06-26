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

use std::collections::HashMap;
use std::io::{Write, Cursor};
use std::path::Path;

// ─── Magic & Constants ──────────────────────────────────

pub const TAAS_MAGIC: &[u8; 5] = b"TAAS1";
pub const TAAS_VERSION: &str = "0.1.0";

/// Тип данных тензора.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    F32,
    I32,
    I64,
    U64,
}

impl TensorDtype {
    pub fn as_str(&self) -> &'static str {
        match self { TensorDtype::F32 => "f32", TensorDtype::I32 => "i32", TensorDtype::I64 => "i64", TensorDtype::U64 => "u64" }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        match s { "f32" => Some(Self::F32), "i32" => Some(Self::I32), "i64" => Some(Self::I64), "u64" => Some(Self::U64), _ => None }
    }
    pub fn elem_size(&self) -> usize {
        match self { Self::F32 => 4, Self::I32 => 4, Self::I64 => 8, Self::U64 => 8 }
    }
}

/// Описание одного тензора/компонента в манифесте.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub component_type: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: u64,
}

/// JSON манифест.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaasManifest {
    pub version: String,
    pub created: String,
    pub components: Vec<ComponentEntry>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

// ─── Writer ─────────────────────────────────────────────

/// Строит .taas файл: собирает тензоры в JSON манифест + бинарные данные.
pub struct TaasWriter {
    components: Vec<ComponentEntry>,
    tensor_data: Vec<u8>,
    metadata: HashMap<String, String>,
}

impl TaasWriter {
    pub fn new() -> Self {
        Self { components: Vec::new(), tensor_data: Vec::new(), metadata: HashMap::new() }
    }

    /// Добавить метаданные (например "description", "model", "dt", "n_neurons").
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Добавить тензор f32.
    pub fn add_f32(&mut self, name: &str, component_type: &str, data: &[f32]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        let shape = vec![data.len()];
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "f32".to_string(),
            shape,
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить тензор f32 с multidimensional shape.
    pub fn add_f32_shape(&mut self, name: &str, component_type: &str, data: &[f32], shape: Vec<usize>) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "f32".to_string(),
            shape,
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить тензор i32.
    pub fn add_i32(&mut self, name: &str, component_type: &str, data: &[i32]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "i32".to_string(),
            shape: vec![data.len()],
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить тензор i32 с multidimensional shape.
    pub fn add_i32_shape(&mut self, name: &str, component_type: &str, data: &[i32], shape: Vec<usize>) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 4) as u64;
        let total: usize = shape.iter().product();
        assert_eq!(total, data.len(), "Shape {:?} doesn't match data len {}", shape, data.len());
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "i32".to_string(),
            shape,
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить тензор i64.
    pub fn add_i64(&mut self, name: &str, component_type: &str, data: &[i64]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 8) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "i64".to_string(),
            shape: vec![data.len()],
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить тензор u64.
    pub fn add_u64(&mut self, name: &str, component_type: &str, data: &[u64]) {
        let offset = self.tensor_data.len() as u64;
        let size = (data.len() * 8) as u64;
        self.components.push(ComponentEntry {
            name: name.to_string(),
            component_type: component_type.to_string(),
            dtype: "u64".to_string(),
            shape: vec![data.len()],
            offset,
            size,
        });
        self.tensor_data.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Добавить одну f32 как тензор [1].
    pub fn add_scalar_f32(&mut self, name: &str, component_type: &str, value: f32) {
        self.add_f32(name, component_type, &[value]);
    }

    /// Добавить одну bool как i32 [1].
    pub fn add_scalar_bool(&mut self, name: &str, component_type: &str, value: bool) {
        self.add_i32(name, component_type, &[if value { 1 } else { 0 }]);
    }

    /// Записать .taas файл на диск.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), String> {
        let manifest = TaasManifest {
            version: TAAS_VERSION.to_string(),
            created: format!("{}", chrono_or_fallback()),
            components: self.components.clone(),
            metadata: self.metadata.clone(),
        };
        let header_json = serde_json::to_string(&manifest).map_err(|e| format!("Failed to serialize manifest: {e}"))?;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Write: magic + header_len + header + tensor_data
        writer.write_all(TAAS_MAGIC).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(&header_len.to_le_bytes()).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(header_bytes).map_err(|e| format!("Write error: {e}"))?;
        writer.write_all(&self.tensor_data).map_err(|e| format!("Write error: {e}"))?;

        Ok(())
    }

    /// Записать в файл.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let mut file = std::fs::File::create(path.as_ref()).map_err(|e| format!("Cannot create file: {e}"))?;
        self.save(&mut file)
    }

    /// Получить байты .taas файла.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        self.save(&mut Cursor::new(&mut buf))?;
        Ok(buf)
    }
}

impl Default for TaasWriter {
    fn default() -> Self { Self::new() }
}

// ─── Reader ─────────────────────────────────────────────

/// Читает .taas файл: манифест + доступ к тензорам по смещению.
pub struct TaasReader {
    pub manifest: TaasManifest,
    data: Vec<u8>,
    data_offset: usize, // начало tensor_data в self.data
}

impl TaasReader {
    /// Загрузить .taas из bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;

        // Magic
        if bytes.len() < 5 + 8 {
            return Err("File too short".to_string());
        }
        if &bytes[0..5] != TAAS_MAGIC {
            return Err(format!("Invalid magic: expected TAAS1, got {:?}", &bytes[0..5]));
        }
        pos += 5;

        // Header length (u64 LE)
        let header_len = u64::from_le_bytes(
            bytes[pos..pos+8].try_into().map_err(|_| "Header length field truncated".to_string())?
        ) as usize;
        pos += 8;

        // Header JSON
        if pos + header_len > bytes.len() {
            return Err("Header truncated".to_string());
        }
        let header_str = std::str::from_utf8(&bytes[pos..pos+header_len])
            .map_err(|e| format!("Invalid UTF-8 in header: {e}"))?;
        let manifest: TaasManifest = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid manifest JSON: {e}"))?;
        pos += header_len;

        // Tensor data starts here
        let data_offset = pos;
        let data = bytes.to_vec();

        Ok(Self { manifest, data, data_offset })
    }

    /// Загрузить из файла.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| format!("Cannot read file: {e}"))?;
        Self::from_bytes(&bytes)
    }

    /// Получить f32 тензор по имени компонента (возвращает копию — безопасно для unaligned data).
    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "f32" {
            return Err(format!("Component '{name}' is not f32, got {}", entry.dtype));
        }
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(format!("Component '{name}' data out of bounds"));
        }
        let n = entry.size as usize / 4;
        let mut result = vec![0.0f32; n];
        let src = &self.data[start..end];
        for i in 0..n {
            let off = i * 4;
            result[i] = f32::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
            ]);
        }
        Ok(result)
    }

    /// Получить i32 тензор по имени компонента.
    pub fn get_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "i32" {
            return Err(format!("Component '{name}' is not i32, got {}", entry.dtype));
        }
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(format!("Component '{name}' data out of bounds"));
        }
        let n = entry.size as usize / 4;
        let mut result = vec![0i32; n];
        let src = &self.data[start..end];
        for i in 0..n {
            let off = i * 4;
            result[i] = i32::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
            ]);
        }
        Ok(result)
    }

    /// Получить i64 тензор по имени компонента.
    pub fn get_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "i64" {
            return Err(format!("Component '{name}' is not i64, got {}", entry.dtype));
        }
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(format!("Component '{name}' data out of bounds"));
        }
        let n = entry.size as usize / 8;
        let mut result = vec![0i64; n];
        let src = &self.data[start..end];
        for i in 0..n {
            let off = i * 8;
            result[i] = i64::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
                src[off+4], src[off+5], src[off+6], src[off+7],
            ]);
        }
        Ok(result)
    }

    /// Получить u64 тензор по имени компонента.
    pub fn get_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        let entry = self.find_component(name)?;
        if entry.dtype != "u64" {
            return Err(format!("Component '{name}' is not u64, got {}", entry.dtype));
        }
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(format!("Component '{name}' data out of bounds"));
        }
        let n = entry.size as usize / 8;
        let mut result = vec![0u64; n];
        let src = &self.data[start..end];
        for i in 0..n {
            let off = i * 8;
            result[i] = u64::from_le_bytes([
                src[off], src[off+1], src[off+2], src[off+3],
                src[off+4], src[off+5], src[off+6], src[off+7],
            ]);
        }
        Ok(result)
    }

    /// Получить f32 скаляр.
    pub fn get_scalar_f32(&self, name: &str) -> Result<f32, String> {
        let slice = self.get_f32(name)?;
        if slice.len() != 1 {
            return Err(format!("Component '{name}' is not a scalar f32, len={}", slice.len()));
        }
        Ok(slice[0])
    }

    /// Получить bool скаляр.
    pub fn get_scalar_bool(&self, name: &str) -> Result<bool, String> {
        let slice = self.get_i32(name)?;
        if slice.len() != 1 {
            return Err(format!("Component '{name}' is not a scalar bool, len={}", slice.len()));
        }
        Ok(slice[0] != 0)
    }

    /// Найти компонент по имени.
    pub fn find_component(&self, name: &str) -> Result<&ComponentEntry, String> {
        self.manifest.components.iter()
            .find(|c| c.name == name)
            .ok_or_else(|| format!("Component '{name}' not found in manifest"))
    }

    /// Список всех компонентов.
    pub fn list_components(&self) -> &[ComponentEntry] {
        &self.manifest.components
    }

    /// Получить метаданные.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.manifest.metadata
    }

    /// Проверить валидность всех смещений.
    pub fn validate(&self) -> Result<(), String> {
        for comp in &self.manifest.components {
            let start = comp.offset as usize;
            let end = start + comp.size as usize;
            if end > self.data.len() - self.data_offset {
                return Err(format!("Component '{}' (offset={}, size={}) exceeds data (len={})",
                    comp.name, comp.offset, comp.size, self.data.len() - self.data_offset));
            }
            // Check alignment
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

// ─── Fallback timestamp (without chrono dependency) ─────

fn chrono_or_fallback() -> String {
    // Try to get ISO timestamp without chrono dep
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple UTC breakdown
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs_rem = time_secs % 60;

    // Days since epoch → year (approximate)
    let year = 1970 + (days as f64 / 365.25) as u64;
    format!("{year}-01-01T{hours:02}:{mins:02}:{secs_rem:02}Z")
}

// ─── Component serialization helpers ───────────────────

/// Снапшот для TriOscillator.
pub fn save_trioscilator(tri: &crate::t3::TriOscillator, writer: &mut TaasWriter) {
    writer.add_metadata("type", "TriOscillator");
    writer.add_metadata("n", &tri.n.to_string());
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
    writer.add_f32_shape("cross_w12", "TriOscillator", &tri.cross_w12, vec![tri.n, tri.n]);
    writer.add_f32_shape("cross_w13", "TriOscillator", &tri.cross_w13, vec![tri.n, tri.n]);
    writer.add_f32_shape("cross_w23", "TriOscillator", &tri.cross_w23, vec![tri.n, tri.n]);
    writer.add_scalar_f32("alpha", "TriOscillator", tri.alpha);
    writer.add_scalar_f32("drive_gain", "TriOscillator", tri.drive_gain);
    writer.add_scalar_f32("epsilon", "TriOscillator", tri.epsilon);
    writer.add_scalar_f32("phase_gate_gain", "TriOscillator", tri.phase_gate_gain);
    writer.add_scalar_f32("gamma1", "TriOscillator", tri.gamma1);
    writer.add_scalar_f32("gamma2", "TriOscillator", tri.gamma2);
    writer.add_scalar_f32("gamma3", "TriOscillator", tri.gamma3);
    writer.add_scalar_bool("initialised", "TriOscillator", tri.initialised);
}

/// Загрузить TriOscillator из reader.
pub fn load_trioscilator(reader: &TaasReader, n: usize) -> Result<crate::t3::TriOscillator<crate::t3::Avx2Backend>, String> {
    use crate::t3::{TriOscillator, Avx2Backend};
    // Create with dummy values, then overwrite
    let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

    tri.th1.copy_from_slice(&reader.get_f32("th1")?);
    tri.th2.copy_from_slice(&reader.get_f32("th2")?);
    tri.th3.copy_from_slice(&reader.get_f32("th3")?);
    if let Ok(v) = reader.get_f32("prev_th1") { tri.prev_th1.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("prev_th2") { tri.prev_th2.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("prev_th3") { tri.prev_th3.copy_from_slice(&v); }
    tri.r1.copy_from_slice(&reader.get_f32("r1")?);
    tri.r2.copy_from_slice(&reader.get_f32("r2")?);
    tri.r3.copy_from_slice(&reader.get_f32("r3")?);
    if let Ok(v) = reader.get_f32("prev_r1") { tri.prev_r1.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("prev_r2") { tri.prev_r2.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("prev_r3") { tri.prev_r3.copy_from_slice(&v); }
    tri.omega1.copy_from_slice(&reader.get_f32("omega1")?);
    tri.omega2.copy_from_slice(&reader.get_f32("omega2")?);
    tri.omega3.copy_from_slice(&reader.get_f32("omega3")?);
    if let Ok(v) = reader.get_f32("drive1") { tri.drive1.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("drive2") { tri.drive2.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("drive3") { tri.drive3.copy_from_slice(&v); }
    tri.sin1.copy_from_slice(&reader.get_f32("sin1")?);
    tri.cos1.copy_from_slice(&reader.get_f32("cos1")?);
    tri.sin2.copy_from_slice(&reader.get_f32("sin2")?);
    tri.cos2.copy_from_slice(&reader.get_f32("cos2")?);
    tri.sin3.copy_from_slice(&reader.get_f32("sin3")?);
    tri.cos3.copy_from_slice(&reader.get_f32("cos3")?);
    tri.coupling1.copy_from_slice(&reader.get_f32("coupling1")?);
    tri.coupling2.copy_from_slice(&reader.get_f32("coupling2")?);
    tri.coupling3.copy_from_slice(&reader.get_f32("coupling3")?);
    if let Ok(v) = reader.get_f32("cross_w12") { tri.cross_w12.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("cross_w13") { tri.cross_w13.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("cross_w23") { tri.cross_w23.copy_from_slice(&v); }
    tri.alpha = reader.get_scalar_f32("alpha").unwrap_or(1.0);
    tri.drive_gain = reader.get_scalar_f32("drive_gain").unwrap_or(10.0);
    tri.epsilon = reader.get_scalar_f32("epsilon").unwrap_or(0.1);
    tri.phase_gate_gain = reader.get_scalar_f32("phase_gate_gain").unwrap_or(4.0);
    tri.gamma1 = reader.get_scalar_f32("gamma1").unwrap_or(0.5);
    tri.gamma2 = reader.get_scalar_f32("gamma2").unwrap_or(1.0);
    tri.gamma3 = reader.get_scalar_f32("gamma3").unwrap_or(1.12);
    tri.initialised = reader.get_scalar_bool("initialised").unwrap_or(true);

    Ok(tri)
}

/// Снапшот для NlmsDecoder.
pub fn save_nlms_decoder(dec: &crate::t3::NlmsDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "NlmsDecoder");
    writer.add_f32("nlms_weights", "NlmsDecoder", &dec.weights);
    writer.add_f32("nlms_bias", "NlmsDecoder", &dec.bias);
    writer.add_scalar_f32("nlms_lr", "NlmsDecoder", dec.lr);
    writer.add_scalar_f32("nlms_eps", "NlmsDecoder", dec.eps);
}

/// Загрузить NlmsDecoder.
pub fn load_nlms_decoder(reader: &TaasReader, n: usize) -> Result<crate::t3::NlmsDecoder, String> {
    let mut dec = crate::t3::NlmsDecoder::new(n, 0.01, 1e-8);
    let weights = reader.get_f32("nlms_weights")?;
    let n_w = weights.len().min(dec.weights.len());
    dec.weights[..n_w].copy_from_slice(&weights[..n_w]);
    if let Ok(bias) = reader.get_f32("nlms_bias") {
        if !bias.is_empty() { dec.bias[0] = bias[0]; }
    }
    dec.lr = reader.get_scalar_f32("nlms_lr").unwrap_or(0.01);
    dec.eps = reader.get_scalar_f32("nlms_eps").unwrap_or(1e-8);
    Ok(dec)
}

/// Снапшот для KalmanDecoder.
pub fn save_kalman_decoder(dec: &crate::t3::KalmanDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "KalmanDecoder");
    writer.add_f32("kalman_weights", "KalmanDecoder", &dec.weights);
    writer.add_f32("kalman_cov", "KalmanDecoder", &dec.cov);
    writer.add_scalar_f32("kalman_bias", "KalmanDecoder", dec.bias);
    writer.add_scalar_f32("kalman_q", "KalmanDecoder", dec.q);
    writer.add_scalar_f32("kalman_r", "KalmanDecoder", dec.r);
}

/// Загрузить KalmanDecoder.
pub fn load_kalman_decoder(reader: &TaasReader, n: usize) -> Result<crate::t3::KalmanDecoder, String> {
    let mut dec = crate::t3::KalmanDecoder::new(n, 0.001, 0.1);
    let weights = reader.get_f32("kalman_weights")?;
    let n_w = weights.len().min(dec.weights.len());
    dec.weights[..n_w].copy_from_slice(&weights[..n_w]);
    if let Ok(cov) = reader.get_f32("kalman_cov") {
        let n_c = cov.len().min(dec.cov.len());
        dec.cov[..n_c].copy_from_slice(&cov[..n_c]);
    }
    dec.bias = reader.get_scalar_f32("kalman_bias").unwrap_or(0.0);
    dec.q = reader.get_scalar_f32("kalman_q").unwrap_or(0.001);
    dec.r = reader.get_scalar_f32("kalman_r").unwrap_or(0.1);
    Ok(dec)
}

/// Снапшот для SornNetwork.
pub fn save_sorn_network(sorn: &crate::models::neuron::SornNetwork, writer: &mut TaasWriter) {
    writer.add_metadata("type", "SornNetwork");
    writer.add_metadata("n", &sorn.n.to_string());
    let n = sorn.n;
    writer.add_f32_shape("sorn_weights", "SornNetwork", &sorn.weights, vec![n, n]);
    writer.add_f32("sorn_theta", "SornNetwork", &sorn.theta);
    writer.add_f32("sorn_rates", "SornNetwork", &sorn.rates);
    writer.add_scalar_f32("sorn_lr_stdp_ltd", "SornNetwork", sorn.lr_stdp_ltd);
    writer.add_scalar_f32("sorn_lr_stdp_ltp", "SornNetwork", sorn.lr_stdp_ltp);
    writer.add_scalar_f32("sorn_w_min", "SornNetwork", sorn.w_min);
    writer.add_scalar_f32("sorn_w_max", "SornNetwork", sorn.w_max);
    writer.add_scalar_f32("sorn_w_target_sum", "SornNetwork", sorn.w_target_sum);
    writer.add_scalar_f32("sorn_lr_ip", "SornNetwork", sorn.lr_ip);
    writer.add_scalar_f32("sorn_target_rate", "SornNetwork", sorn.target_rate);
    writer.add_scalar_f32("sorn_ip_decay", "SornNetwork", sorn.ip_decay);
}

/// Загрузить SornNetwork.
pub fn load_sorn_network(reader: &TaasReader, n: usize) -> Result<crate::models::neuron::SornNetwork, String> {
    let mut sorn = crate::models::neuron::SornNetwork::new(n, 0.1, 10.0);
    if let Ok(w) = reader.get_f32("sorn_weights") {
        let n_w = w.len().min(sorn.weights.len());
        sorn.weights[..n_w].copy_from_slice(&w[..n_w]);
    }
    if let Ok(th) = reader.get_f32("sorn_theta") {
        let n_t = th.len().min(sorn.theta.len());
        sorn.theta[..n_t].copy_from_slice(&th[..n_t]);
    }
    if let Ok(r) = reader.get_f32("sorn_rates") {
        let n_r = r.len().min(sorn.rates.len());
        sorn.rates[..n_r].copy_from_slice(&r[..n_r]);
    }
    sorn.lr_stdp_ltd = reader.get_scalar_f32("sorn_lr_stdp_ltd").unwrap_or(0.01);
    sorn.lr_stdp_ltp = reader.get_scalar_f32("sorn_lr_stdp_ltp").unwrap_or(0.005);
    sorn.w_min = reader.get_scalar_f32("sorn_w_min").unwrap_or(0.0);
    sorn.w_max = reader.get_scalar_f32("sorn_w_max").unwrap_or(2.0);
    sorn.w_target_sum = reader.get_scalar_f32("sorn_w_target_sum").unwrap_or(10.0);
    sorn.lr_ip = reader.get_scalar_f32("sorn_lr_ip").unwrap_or(0.001);
    sorn.target_rate = reader.get_scalar_f32("sorn_target_rate").unwrap_or(0.01);
    sorn.ip_decay = reader.get_scalar_f32("sorn_ip_decay").unwrap_or(0.999);
    Ok(sorn)
}

/// Снапшот для EPropLIF.
pub fn save_eprop_lif(eprop: &crate::models::neuron::EPropLIF, writer: &mut TaasWriter) {
    writer.add_metadata("type", "EPropLIF");
    writer.add_metadata("n", &eprop.n.to_string());
    writer.add_f32("eprop_v", "EPropLIF", &eprop.v);
    writer.add_f32("eprop_theta", "EPropLIF", &eprop.theta);
    writer.add_f32("eprop_eligibility", "EPropLIF", &eprop.eligibility);
    writer.add_f32("eprop_psi", "EPropLIF", &eprop.psi);
    writer.add_scalar_f32("eprop_dt", "EPropLIF", eprop.dt);
    writer.add_scalar_f32("eprop_tau_m", "EPropLIF", eprop.tau_m);
    writer.add_scalar_f32("eprop_tau_e", "EPropLIF", eprop.tau_e);
    writer.add_scalar_f32("eprop_v_rest", "EPropLIF", eprop.v_rest);
    writer.add_scalar_f32("eprop_v_thresh", "EPropLIF", eprop.v_thresh);
    writer.add_scalar_f32("eprop_v_reset", "EPropLIF", eprop.v_reset);
    writer.add_scalar_f32("eprop_tau_surrogate", "EPropLIF", eprop.tau_surrogate);
}

/// Загрузить EPropLIF.
pub fn load_eprop_lif(reader: &TaasReader, n: usize) -> Result<crate::models::neuron::EPropLIF, String> {
    let mut eprop = crate::models::neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    if let Ok(v) = reader.get_f32("eprop_v") { eprop.v.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("eprop_theta") { eprop.theta.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("eprop_eligibility") { eprop.eligibility.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("eprop_psi") { eprop.psi.copy_from_slice(&v); }
    eprop.dt = reader.get_scalar_f32("eprop_dt").unwrap_or(1.0);
    eprop.tau_m = reader.get_scalar_f32("eprop_tau_m").unwrap_or(10.0);
    eprop.tau_e = reader.get_scalar_f32("eprop_tau_e").unwrap_or(20.0);
    eprop.v_rest = reader.get_scalar_f32("eprop_v_rest").unwrap_or(-65.0);
    eprop.v_thresh = reader.get_scalar_f32("eprop_v_thresh").unwrap_or(-50.0);
    eprop.v_reset = reader.get_scalar_f32("eprop_v_reset").unwrap_or(-70.0);
    eprop.tau_surrogate = reader.get_scalar_f32("eprop_tau_surrogate").unwrap_or(2.0);
    Ok(eprop)
}

/// Снапшот для ShortTermPlasticity.
pub fn save_stp(stp: &crate::models::neuron::ShortTermPlasticity, writer: &mut TaasWriter) {
    writer.add_metadata("type", "ShortTermPlasticity");
    writer.add_metadata("n", &stp.n.to_string());
    writer.add_f32("stp_x", "ShortTermPlasticity", &stp.x);
    writer.add_f32("stp_u", "ShortTermPlasticity", &stp.u);
    writer.add_f32("stp_tau_d", "ShortTermPlasticity", &stp.tau_d);
    writer.add_f32("stp_tau_f", "ShortTermPlasticity", &stp.tau_f);
    writer.add_f32("stp_u0", "ShortTermPlasticity", &stp.u0);
    writer.add_f32("stp_a", "ShortTermPlasticity", &stp.a);
    writer.add_scalar_f32("stp_dt", "ShortTermPlasticity", stp.dt);
}

/// Загрузить ShortTermPlasticity.
pub fn load_stp(reader: &TaasReader, n: usize) -> Result<crate::models::neuron::ShortTermPlasticity, String> {
    let mut stp = crate::models::neuron::ShortTermPlasticity::new(n, 1.0);
    if let Ok(v) = reader.get_f32("stp_x") { stp.x.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("stp_u") { stp.u.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("stp_tau_d") { stp.tau_d.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("stp_tau_f") { stp.tau_f.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("stp_u0") { stp.u0.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("stp_a") { stp.a.copy_from_slice(&v); }
    stp.dt = reader.get_scalar_f32("stp_dt").unwrap_or(1.0);
    Ok(stp)
}

/// Снапшот для ActorCriticAgent (веса + eligibility + STDP traces + параметры).
/// Сохраняет полное состояние обучения, чтобы можно было восстановить агента.
pub fn save_actor_critic_agent(agent: &crate::training::rl::ActorCriticAgent, writer: &mut TaasWriter) {
    writer.add_metadata("type", "ActorCriticAgent");
    writer.add_metadata("n_input", &agent.n_input.to_string());
    writer.add_metadata("n_hidden", &agent.n_hidden.to_string());
    writer.add_metadata("n_actions", &agent.n_actions_real.to_string());
    writer.add_metadata("n_steps", &agent.n_steps.to_string());

    // Weights
    writer.add_f32("rl_w_ih", "ActorCriticAgent", &agent.w_ih);
    writer.add_f32("rl_w_ho", "ActorCriticAgent", &agent.w_ho);
    writer.add_f32("rl_w_ic", "ActorCriticAgent", &agent.w_ic);

    // Eligibility traces
    writer.add_f32("rl_elig_ih", "ActorCriticAgent", &agent.elig_ih);
    writer.add_f32("rl_elig_ho", "ActorCriticAgent", &agent.elig_ho);
    writer.add_f32("rl_elig_ic", "ActorCriticAgent", &agent.elig_ic);

    // STDP traces
    writer.add_f32("rl_trace_input", "ActorCriticAgent", &agent.trace_input);
    writer.add_f32("rl_trace_hidden", "ActorCriticAgent", &agent.trace_hidden);
    writer.add_f32("rl_trace_output", "ActorCriticAgent", &agent.trace_output);
    writer.add_f32("rl_trace_critic", "ActorCriticAgent", &agent.trace_critic_val);

    // Learning params
    writer.add_scalar_f32("rl_dt", "ActorCriticAgent", agent.dt);
    writer.add_scalar_f32("rl_gamma", "ActorCriticAgent", agent.gamma);
    writer.add_scalar_f32("rl_lr_actor", "ActorCriticAgent", agent.lr_actor);
    writer.add_scalar_f32("rl_lr_critic", "ActorCriticAgent", agent.lr_critic);
    writer.add_scalar_f32("rl_tau_trace", "ActorCriticAgent", agent.tau_trace);
    writer.add_scalar_f32("rl_tau_e", "ActorCriticAgent", agent.tau_e);
    writer.add_scalar_f32("rl_a_ltp", "ActorCriticAgent", agent.a_ltp);
    writer.add_scalar_f32("rl_a_ltd", "ActorCriticAgent", agent.a_ltd);
    writer.add_scalar_f32("rl_w_min", "ActorCriticAgent", agent.w_min);
    writer.add_scalar_f32("rl_w_max", "ActorCriticAgent", agent.w_max);
    writer.add_scalar_f32("rl_tau_ref", "ActorCriticAgent", agent.tau_ref);
    writer.add_scalar_f32("rl_last_value", "ActorCriticAgent", agent.last_value);
}

/// Загрузить ActorCriticAgent из .taas reader.
/// Восстанавливает: веса, eligibility, STDP traces, все гиперпараметры.
pub fn load_actor_critic_agent(reader: &TaasReader) -> Result<crate::training::rl::ActorCriticAgent, String> {
    let n_input = reader.metadata().get("n_input")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_input not found in metadata".to_string())?;
    let n_hidden = reader.metadata().get("n_hidden")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_hidden not found in metadata".to_string())?;
    let n_actions = reader.metadata().get("n_actions")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_actions not found in metadata".to_string())?;
    let n_steps = reader.metadata().get("n_steps")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_steps not found in metadata".to_string())?;

    let dt = reader.get_scalar_f32("rl_dt").unwrap_or(1.0);
    let mut agent = crate::training::rl::ActorCriticAgent::new(n_input, n_hidden, n_actions, n_steps, dt, 42);

    // Restore weights
    macro_rules! restore_f32 {
        ($agent:ident, $reader:ident, $field:ident, $name:expr) => {
            if let Ok(v) = $reader.get_f32($name) {
                let n = v.len().min($agent.$field.len());
                $agent.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }

    restore_f32!(agent, reader, w_ih, "rl_w_ih");
    restore_f32!(agent, reader, w_ho, "rl_w_ho");
    restore_f32!(agent, reader, w_ic, "rl_w_ic");

    restore_f32!(agent, reader, elig_ih, "rl_elig_ih");
    restore_f32!(agent, reader, elig_ho, "rl_elig_ho");
    restore_f32!(agent, reader, elig_ic, "rl_elig_ic");

    restore_f32!(agent, reader, trace_input, "rl_trace_input");
    restore_f32!(agent, reader, trace_hidden, "rl_trace_hidden");
    restore_f32!(agent, reader, trace_output, "rl_trace_output");
    restore_f32!(agent, reader, trace_critic_val, "rl_trace_critic");

    // Restore params
    agent.gamma = reader.get_scalar_f32("rl_gamma").unwrap_or(0.95);
    agent.lr_actor = reader.get_scalar_f32("rl_lr_actor").unwrap_or(0.01);
    agent.lr_critic = reader.get_scalar_f32("rl_lr_critic").unwrap_or(0.01);
    agent.tau_trace = reader.get_scalar_f32("rl_tau_trace").unwrap_or(20.0);
    agent.tau_e = reader.get_scalar_f32("rl_tau_e").unwrap_or(500.0);
    agent.a_ltp = reader.get_scalar_f32("rl_a_ltp").unwrap_or(0.1);
    agent.a_ltd = reader.get_scalar_f32("rl_a_ltd").unwrap_or(0.12);
    agent.w_min = reader.get_scalar_f32("rl_w_min").unwrap_or(0.0);
    agent.w_max = reader.get_scalar_f32("rl_w_max").unwrap_or(2.0);
    agent.tau_ref = reader.get_scalar_f32("rl_tau_ref").unwrap_or(3.0);
    agent.last_value = reader.get_scalar_f32("rl_last_value").unwrap_or(0.0);

    Ok(agent)
}

/// Снапшот для PoissonGenerator (rates + dt).
pub fn save_poisson(poisson: &crate::models::neuron::PoissonGenerator, writer: &mut TaasWriter) {
    writer.add_metadata("type", "PoissonGenerator");
    writer.add_metadata("n", &poisson.n_neurons().to_string());
    writer.add_f32("poisson_rates", "PoissonGenerator", poisson.rates());
    writer.add_scalar_f32("poisson_dt", "PoissonGenerator", poisson.dt());
}

/// Снапшот для TTFS encoder state.
pub fn save_ttfs(ttfs: &crate::models::neuron::TtfsEncoder, writer: &mut TaasWriter, _n: usize) {
    writer.add_metadata("type", "TtfsEncoder");
    let fire_times: Vec<u64> = ttfs.fire_times.iter().map(|&x| x as u64).collect();
    let fired: Vec<i32> = ttfs.fired.iter().map(|&x| if x { 1 } else { 0 }).collect();
    let step = [ttfs.current_step as u64];
    let delay = [ttfs.max_delay as u64];
    writer.add_u64("ttfs_fire_times", "TtfsEncoder", &fire_times);
    writer.add_i32("ttfs_fired", "TtfsEncoder", &fired);
    writer.add_u64("ttfs_current_step", "TtfsEncoder", &step);
    writer.add_u64("ttfs_max_delay", "TtfsEncoder", &delay);
}

/// Снапшот для LIF-AT neuron state (v, theta).
pub fn save_lif_at_state(v: &[f32], theta: &[f32], n: usize, writer: &mut TaasWriter) {
    let n_save = n.min(v.len());
    writer.add_metadata("type", "LifNeuronState");
    writer.add_f32("lif_v", "LifNeuronState", &v[..n_save]);
    writer.add_f32("lif_theta", "LifNeuronState", &theta[..n_save]);
}

/// STDP weights (plain vector).
pub fn save_stdp_weights(weights: &[f32], n: usize, writer: &mut TaasWriter) {
    writer.add_metadata("type", "STDPWeights");
    let n_save = n.min(weights.len());
    writer.add_f32("stdp_weights", "STDPWeights", &weights[..n_save]);
}

/// CSR sparse matrix save.
pub fn save_csr(row_ptr: &[i32], col_indices: &[i32], values: &[f32], n_neurons: usize, writer: &mut TaasWriter) {
    writer.add_metadata("type", "CSRMatrix");
    writer.add_metadata("n_neurons", &n_neurons.to_string());
    writer.add_i32("csr_row_ptr", "CSRMatrix", row_ptr);
    writer.add_i32("csr_col_indices", "CSRMatrix", col_indices);
    writer.add_f32("csr_values", "CSRMatrix", values);
}

// ─── Save/Load for remaining decoders ─────────────────────

/// Снапшот для InverseKuramotoDecoder.
pub fn save_inverse_kuramoto(dec: &crate::t3::InverseKuramotoDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "InverseKuramotoDecoder");
    writer.add_metadata("n", &dec.n.to_string());
    writer.add_f32("ikd_prev_amps", "InverseKuramotoDecoder", &dec.prev_amps);
    writer.add_f32("ikd_weights", "InverseKuramotoDecoder", &dec.weights);
    writer.add_scalar_f32("ikd_alpha", "InverseKuramotoDecoder", dec.alpha);
    writer.add_scalar_f32("ikd_drive_gain", "InverseKuramotoDecoder", dec.drive_gain);
    writer.add_scalar_f32("ikd_epsilon", "InverseKuramotoDecoder", dec.epsilon);
    writer.add_scalar_bool("ikd_initialised", "InverseKuramotoDecoder", dec.initialised);
}

/// Загрузить InverseKuramotoDecoder.
pub fn load_inverse_kuramoto(reader: &TaasReader, n: usize) -> Result<crate::t3::InverseKuramotoDecoder, String> {
    let mut dec = crate::t3::InverseKuramotoDecoder::new(n, 1.0, 1.0);
    if let Ok(v) = reader.get_f32("ikd_prev_amps") { dec.prev_amps.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("ikd_weights") { dec.weights.copy_from_slice(&v); }
    dec.alpha = reader.get_scalar_f32("ikd_alpha").unwrap_or(1.0);
    dec.drive_gain = reader.get_scalar_f32("ikd_drive_gain").unwrap_or(1.0);
    dec.epsilon = reader.get_scalar_f32("ikd_epsilon").unwrap_or(0.1);
    dec.initialised = reader.get_scalar_bool("ikd_initialised").unwrap_or(false);
    Ok(dec)
}

/// Снапшот для StdpDecoder.
pub fn save_stdp_decoder(dec: &crate::t3::StdpDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "StdpDecoder");
    writer.add_metadata("n", &dec.weights.len().to_string());
    writer.add_f32("sd_weights", "StdpDecoder", &dec.weights);
    writer.add_f32("sd_traces", "StdpDecoder", &dec.traces);
    writer.add_f32("sd_prev_states", "StdpDecoder", &dec.prev_states);
    writer.add_scalar_f32("sd_bias", "StdpDecoder", dec.bias);
    writer.add_scalar_f32("sd_lr", "StdpDecoder", dec.lr);
    writer.add_scalar_f32("sd_decay", "StdpDecoder", dec.decay);
    writer.add_scalar_f32("sd_sparsity", "StdpDecoder", dec.sparsity);
}

/// Загрузить StdpDecoder.
pub fn load_stdp_decoder(reader: &TaasReader, n: usize) -> Result<crate::t3::StdpDecoder, String> {
    let mut dec = crate::t3::StdpDecoder::new(n, 0.01, 0.9, 1e-5);
    if let Ok(v) = reader.get_f32("sd_weights") { dec.weights.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("sd_traces") { dec.traces.copy_from_slice(&v); }
    if let Ok(v) = reader.get_f32("sd_prev_states") { dec.prev_states.copy_from_slice(&v); }
    dec.bias = reader.get_scalar_f32("sd_bias").unwrap_or(0.0);
    dec.lr = reader.get_scalar_f32("sd_lr").unwrap_or(0.01);
    dec.decay = reader.get_scalar_f32("sd_decay").unwrap_or(0.9);
    dec.sparsity = reader.get_scalar_f32("sd_sparsity").unwrap_or(1e-5);
    dec.initialised = reader.get_scalar_bool("sd_initialised").unwrap_or(false);
    Ok(dec)
}

/// Снапшот для PstDecoder.
pub fn save_pst_decoder(dec: &crate::t3::PstDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "PstDecoder");
    writer.add_metadata("feat_dim", &dec.feat_dim.to_string());
    writer.add_metadata("L", &dec.L.to_string());
    writer.add_f32("pst_weights", "PstDecoder", &dec.weights);
    writer.add_scalar_f32("pst_bias", "PstDecoder", dec.bias);
    writer.add_scalar_f32("pst_alpha", "PstDecoder", dec.alpha);
}

/// Загрузить PstDecoder.
pub fn load_pst_decoder(reader: &TaasReader, feat_dim: usize, l: usize) -> Result<crate::t3::PstDecoder, String> {
    let mut dec = crate::t3::PstDecoder::new(feat_dim, l, 0.01);
    if let Ok(v) = reader.get_f32("pst_weights") { dec.weights.copy_from_slice(&v); }
    dec.bias = reader.get_scalar_f32("pst_bias").unwrap_or(0.0);
    dec.alpha = reader.get_scalar_f32("pst_alpha").unwrap_or(0.01);
    Ok(dec)
}

// ─── Load helpers for state-only types ────────────────────

/// Загрузить LIF-AT состояние (v, theta).
pub fn load_lif_at_state(reader: &TaasReader) -> Result<(Vec<f32>, Vec<f32>), String> {
    let v = reader.get_f32("lif_v")?;
    let theta = reader.get_f32("lif_theta")?;
    Ok((v, theta))
}

/// Загрузить STDP веса.
pub fn load_stdp_weights(reader: &TaasReader) -> Result<Vec<f32>, String> {
    reader.get_f32("stdp_weights")
}

/// Загрузить CSR матрицу.
pub fn load_csr(reader: &TaasReader) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), String> {
    let row_ptr = reader.get_i32("csr_row_ptr")?;
    let col_indices = reader.get_i32("csr_col_indices")?;
    let values = reader.get_f32("csr_values")?;
    Ok((row_ptr, col_indices, values))
}

// ─── Utility: inspect .taas file ─────────────────────────

/// Распечатать содержимое .taas файла для CLI `taas inspect`.
pub fn inspect_taas(path: &str) -> Result<String, String> {
    let reader = TaasReader::from_file(path)?;
    let mut lines = Vec::new();

    lines.push(format!("📦 .taas archive: {}", path));
    lines.push(format!("   Version: {}", reader.manifest.version));
    lines.push(format!("   Created: {}", reader.manifest.created));
    lines.push(format!("   Components: {}", reader.manifest.components.len()));
    lines.push(String::new());

    for comp in &reader.manifest.components {
        let n_elems: usize = comp.shape.iter().product();
        let size_str = if comp.size > 1024 * 1024 {
            format!("{:.1} MB", comp.size as f64 / (1024.0 * 1024.0))
        } else if comp.size > 1024 {
            format!("{:.1} KB", comp.size as f64 / 1024.0)
        } else {
            format!("{} B", comp.size)
        };
        lines.push(format!("   📄 {:<25} {}  {:>8} {} {}",
            comp.name, comp.component_type, size_str, comp.dtype, format!("{:?}", comp.shape)));
        lines.push(format!("   {:>29} offset={}  n_elems={}", "", comp.offset, n_elems));
    }

    if !reader.manifest.metadata.is_empty() {
        lines.push(String::new());
        lines.push("   Metadata:".to_string());
        for (k, v) in &reader.manifest.metadata {
            lines.push(format!("      {:<20} = {}", k, v));
        }
    }

    lines.push(String::new());
    let total_size = if reader.data.len() > 1024 * 1024 {
        format!("{:.1} MB", reader.data.len() as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB", reader.data.len() as f64 / 1024.0)
    };
    lines.push(format!("   Total size: {}", total_size));

    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t3::Avx2Backend;

    #[test]
    fn test_roundtrip_taas_basic() {
        let mut writer = TaasWriter::new();
        writer.add_f32("test_tensor", "Test", &[1.0, 2.0, 3.0, 4.0]);
        writer.add_i32("test_int", "Test", &[10, 20, 30]);
        writer.add_scalar_f32("test_scalar", "Test", 42.0);
        writer.add_scalar_bool("test_bool", "Test", true);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();

        let f32_data = reader.get_f32("test_tensor").unwrap();
        assert_eq!(f32_data, &[1.0, 2.0, 3.0, 4.0]);

        let i32_data = reader.get_i32("test_int").unwrap();
        assert_eq!(i32_data, &[10, 20, 30]);

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
        use crate::t3::TriOscillator;
        let tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        let mut writer = TaasWriter::new();
        save_trioscilator(&tri, &mut writer);
        let bytes = writer.to_bytes().unwrap();

        // Verify some values match
        let reader = TaasReader::from_bytes(&bytes).unwrap();
        let th1 = reader.get_f32("th1").unwrap();
        assert_eq!(th1.len(), 16);
        assert_eq!(&th1[..], &tri.th1[..]);
    }

    #[test]
    fn test_nlms_decoder_roundtrip() {
        use crate::t3::NlmsDecoder;
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
        use crate::t3::KalmanDecoder;
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
        use crate::models::neuron::SornNetwork;
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
        use crate::t3::{TriOscillator, NlmsDecoder};

        let tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        let dec = NlmsDecoder::new(16, 0.01, 1e-8);

        let mut writer = TaasWriter::new();
        writer.add_metadata("description", "Test multi-component");
        save_trioscilator(&tri, &mut writer);
        save_nlms_decoder(&dec, &mut writer);

        let bytes = writer.to_bytes().unwrap();
        let reader = TaasReader::from_bytes(&bytes).unwrap();

        assert_eq!(reader.manifest.components.len(), 38 + 4); // tri fields (38) + nlms fields (4)
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
        // Save to temp
        let tmp = "/tmp/test_inspect.taas";
        let mut f = std::fs::File::create(tmp).unwrap();
        f.write_all(&bytes).unwrap();
        drop(f);

        let output = inspect_taas(tmp).unwrap();
        assert!(output.contains("test"));
        assert!(output.contains("izhikevich"));
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_load_tri_oscillator() {
        use crate::t3::TriOscillator;
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
}
