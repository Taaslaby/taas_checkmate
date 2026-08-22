//! Эталонные трассы NEURON/Brian2.
//!
//! Встроенные эталоны ([`ReferenceTrace::embedded`]) доступны всегда;
//! загрузка JSON-файлов — за фичей `validate-json` (dep `serde_json`).
//!
//! JSON-форматы:
//!
//! 1. Полный: `{"name": "hh", "t": [0.0, 0.01, ...], "v": [-65.0, ...]}`
//!    (так пишет `resources/validate/neuron_models/gen_*_ref.py`);
//! 2. Компактный: `{"hh": [−65.0, −64.9, ...]}` — только V-временной ряд.

#[cfg(feature = "validate-json")]
use super::ValidationError;

/// Эталонная трасса: V(t) в мВ.
#[derive(Debug, Clone)]
pub struct ReferenceTrace {
    /// Имя эталона (модель/эксперимент).
    pub name: String,
    /// Сетка времени (мс); пустая = равномерная с неизвестным шагом.
    pub t_ms: Vec<f32>,
    /// Напряжение (мВ), той же длины, что `t_ms`.
    pub v_mv: Vec<f32>,
}

impl ReferenceTrace {
    /// Встроенный эталон из констант (без файла).
    pub fn embedded(name: &str, t_ms: &[f32], v_mv: &[f32]) -> Self {
        assert_eq!(t_ms.len(), v_mv.len(), "эталон: t и v одной длины");
        Self {
            name: name.to_string(),
            t_ms: t_ms.to_vec(),
            v_mv: v_mv.to_vec(),
        }
    }
}

/// Загрузить эталоны из JSON-файла (фича `validate-json`).
///
/// Возвращает все распознанные записи «имя → трасса».
#[cfg(feature = "validate-json")]
pub fn load_references(
    path: &std::path::Path,
) -> Result<Vec<ReferenceTrace>, ValidationError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| ValidationError::ReadReference(path.to_path_buf(), err))?;
    let parsed: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| {
            ValidationError::ReadReference(
                path.to_path_buf(),
                std::io::Error::new(std::io::ErrorKind::InvalidData, err),
            )
        })?;
    let Some(obj) = parsed.as_object() else {
        return Ok(Vec::new());
    };

    let mut refs = Vec::new();
    for (key, value) in obj {
        if let Some(trace) = parse_reference(key, value) {
            refs.push(trace);
        }
    }
    Ok(refs)
}

#[cfg(feature = "validate-json")]
fn parse_reference(key: &str, value: &serde_json::Value) -> Option<ReferenceTrace> {
    let Some(obj) = value.as_object() else {
        // Компактный формат: голый массив V.
        let v = value.as_array()?.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
        return Some(ReferenceTrace {
            name: key.to_string(),
            t_ms: Vec::new(),
            v_mv: v,
        });
    };

    let v_mv: Vec<f32> = obj.get("v")?.as_array()?
        .iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
    let t_ms: Vec<f32> = obj.get("t").and_then(|t| t.as_array()).map(|arr| {
        arr.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()
    }).unwrap_or_default();
    if v_mv.is_empty() {
        return None;
    }
    Some(ReferenceTrace {
        name: obj.get("name").and_then(|n| n.as_str()).unwrap_or(key).to_string(),
        t_ms,
        v_mv,
    })
}
