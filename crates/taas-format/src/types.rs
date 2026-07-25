use std::collections::HashMap;

pub const TAAS_MAGIC: &[u8; 5] = b"TAAS1";
pub const TAAS_VERSION: &str = "0.2.0";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    F32,
    I32,
    I64,
    U64,
    U8,
}

impl TensorDtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::U8 => "u8",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "u64" => Some(Self::U64),
            "u8" => Some(Self::U8),
            _ => None,
        }
    }
    pub fn elem_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub component_type: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: u64,
    /// CRC32 checksum of the (decompressed) tensor data (optional).
    /// Old archives won't have this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crc32: Option<u32>,
    /// Compression algorithm, e.g. "zstd". If None, data is uncompressed.
    /// `size` refers to the stored (compressed) size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<String>,
    /// Original uncompressed size (only set when compression is used).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncompressed_size: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaasManifest {
    pub version: String,
    pub created: String,
    pub components: Vec<ComponentEntry>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}
