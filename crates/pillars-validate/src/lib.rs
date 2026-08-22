//! Neuron validation against NEURON reference data

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Validation result for a single neuron model test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Model name (HH, LIF, AdEx, etc.)
    pub name: String,
    /// Maximum absolute voltage difference (mV)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max_diff_mV")]
    pub max_diff_mv: Option<f32>,
    /// Root mean square error (mV)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "rmse_mV")]
    pub rmse_mv: Option<f32>,
    /// Spike count comparison
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spike_count_diff: Option<i64>,
    /// Pass/fail status
    pub passed: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ValidationResult {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_diff_mv: None,
            rmse_mv: None,
            spike_count_diff: None,
            passed: false,
            error: None,
        }
    }

    pub fn calculate_metrics(pillars_v: &[f32], neuron_v: &[f32], tol_mv: f32) -> Self {
        let mut result = Self::new("unnamed");

        // Check lengths match
        if pillars_v.len() != neuron_v.len() {
            result.error = Some(format!(
                "length mismatch: Pillars {} != NEURON {}",
                pillars_v.len(),
                neuron_v.len()
            ));
            return result;
        }

        // Calculate per-point differences
        let mut sum_sq = 0.0f32;
        let mut max_diff: f32 = 0.0;
        let n = pillars_v.len() as f32;

        for (i, (&p, &n)) in pillars_v.iter().zip(neuron_v).enumerate() {
            let diff = (p - n).abs();
            sum_sq += diff * diff;
            max_diff = max_diff.max(diff);

            if diff > tol_mv {
                result.error = Some(format!(
                    "exceeded tolerance at step {}: |{} - {}| = {:.6} mV > {:.1} mV",
                    i, p, n, diff, tol_mv
                ));
            }
        }

        result.rmse_mv = Some((sum_sq / n).sqrt());
        result.max_diff_mv = Some(max_diff);
        result.passed = max_diff <= tol_mv && result.error.is_none();

        result
    }

    pub fn compare_spikes(expected: i64, actual: i64, tol: i64) -> Self {
        let mut result = Self::new("spike-count");
        let diff = (expected - actual).abs();
        
        result.spike_count_diff = Some(actual - expected);
        result.passed = diff <= tol;
        
        if !result.passed {
            result.error = Some(format!(
                "spike count diff too large: expected {}, got {}, diff={}",
                expected, actual, diff
            ));
        }

        result
    }
}

/// Parse NEURON reference results from JSON file
pub fn load_reference_results(json_path: &Path) -> Result<HashMap<String, ReferenceData>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(json_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;
    
    let mut refs = HashMap::new();
    
    if let Some(obj) = parsed.as_object() {
        for (name, value) in obj {
            if let Some(data) = parse_reference_value(value, name) {
                refs.insert(name.clone(), data);
            }
        }
    }
    
    Ok(refs)
}

fn parse_reference_value(value: &serde_json::Value, default_name: &str) -> Option<ReferenceData> {
    // Try object format
    if let Some(obj) = value.as_object() {
        let name = obj.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(default_name)
            .to_string();
            
        let rmse = obj.get("rmse").and_then(|v| v.as_f64()).map(|v| v as f32);
        let v_range = obj.get("v_range").and_then(|v| v.as_f64()).map(|v| v as f32);
        
        return Some(ReferenceData {
            name,
            rmse,
            v_range,
            ..Default::default()
        });
    }
    
    // Try array format (time series data)
    if let Some(arr) = value.as_array() {
        let data = arr.iter()
            .filter_map(|v| v.as_f64())
            .map(|v| v as f32)
            .collect::<Vec<_>>();
            
        return Some(ReferenceData {
            name: default_name.to_string(),
            time_series: Some(data),
            ..Default::default()
        });
    }
    
    None
}

/// Data structure for NEURON reference results
#[derive(Debug, Clone, Default)]
pub struct ReferenceData {
    pub name: String,
    pub rmse: Option<f32>,
    pub v_range: Option<f32>,
    pub spikes: Option<i64>,
    pub time_series: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_basic() {
        let pillars_v = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let neuron_v = vec![0.1, 1.1, 2.1, 3.1, 4.1];
        let tol = 0.5;
        
        let result = ValidationResult::calculate_metrics(&pillars_v, &neuron_v, tol);
        
        assert!(result.passed, "Should pass with small diffs");
        assert!(result.max_diff_mv.unwrap() <= tol);
    }

    #[test]
    fn test_validation_result_fail_tolerance() {
        let pillars_v = vec![0.0, 10.0, 20.0];
        let neuron_v = vec![0.0, 0.0, 0.0];
        let tol = 5.0;
        
        let result = ValidationResult::calculate_metrics(&pillars_v, &neuron_v, tol);
        
        assert!(!result.passed, "Should fail exceeding tolerance");
        assert!(result.error.is_some());
    }

    #[test]
    fn test_spike_count_comparison() {
        let result = ValidationResult::compare_spikes(100, 98, 5);
        assert!(result.passed);
        
        let result_fail = ValidationResult::compare_spikes(100, 110, 5);
        assert!(!result_fail.passed);
    }
}
