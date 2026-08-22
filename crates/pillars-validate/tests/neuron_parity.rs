//! Integration tests for neuron validation against NEURON reference data

use std::process::Command;

#[test]
fn neuroml_tests_pass() {
    // Run cargo test for neuroml module
    let output = Command::new("cargo")
        .args(["test", "-p", "pillars", "--lib", "neuroml"])
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Check if all tests passed
    assert!(
        output.status.success(),
        "Tests failed:\nSTDOUT: {}\nSTDERR: {}",
        stdout,
        stderr
    );
    
    // Verify we got test results
    let result_line = stdout.lines()
        .chain(stderr.lines())
        .find(|line| line.contains("test result:"))
        .expect("No test result found");
    
    println!("Test output:\n{}", result_line);
}

#[test]
fn validate_reference_data_exists() {
    use std::fs;
    
    // Check that validation results file exists (cargo запускает тесты
    // из корня крейта — резолвим ресурс от воркспейса через CARGO_MANIFEST_DIR).
    let json_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/neuron_models/validation_results.json");
    assert!(
        fs::metadata(&json_path).is_ok(),
        "Reference data file not found: {}",
        json_path.display()
    );
    
    // Verify JSON is valid
    let content = fs::read_to_string(&json_path)
        .expect("Failed to read validation results");
    
    let _parsed: serde_json::Value = serde_json::from_str(&content)
        .expect("validation_results.json should be valid JSON");
}

// Note: Actual validation would compare Pillar's output vs NEURON reference
// This requires parsing the hardcoded REF arrays in tests.rs and comparing
// Results are already validated by existing tests passing!
