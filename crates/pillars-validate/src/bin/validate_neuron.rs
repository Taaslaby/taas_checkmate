//! Neuron validation harness comparing Pillars vs NEURON reference data

use anyhow::Result;
use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::process::Command;
use std::time::Instant;

/// CLI tool for validating Pillars neuron models against NEURON reference data
#[derive(Parser, Debug)]
#[command(author, version, about = "Validate Pillars neuron simulations vs NEURON references")]
struct Args {
    /// Verbose output with detailed test results
    #[arg(short, long)]
    verbose: bool,

    /// Number of neurons to simulate (default: use hardcoded test values)
    #[arg(short, long, default_value = "1")]
    n_neurons: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("{}", style("╔══════════════════════════════════════════════════════╗").cyan());
    println!("{}", style("║  Pillars vs NEURON Validation Harness                ║").cyan());
    println!("{}", style("╚══════════════════════════════════════════════════════╝").cyan());
    println!();
    
    let start = Instant::now();
    
    // Step 1: Run the existing neuroml tests
    println!("🔬 Running {} neuroml tests...", count_neuroml_tests()?);
    let progress = ProgressBar::new_spinner();
    progress.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .unwrap()
        .tick_chars("◷◴◳◲◱◰◳◲")
    );
    progress.set_message("Compiling and running tests...");
    progress.enable_steady_tick(std::time::Duration::from_millis(100));
    
    let test_result = run_tests();
    
    progress.finish_and_clear();
    
    let elapsed = start.elapsed();
    
    match test_result {
        Ok(summary) => {
            print_summary(&summary, args.verbose);
            
            if summary.passed == summary.total {
                println!("\n{}", style("✅ All validations passed!").green().bold());
                println!("   Total time: {}", style(elapsed.as_secs_f32()).yellow());
                std::process::exit(0);
            } else {
                println!("\n{}", style("❌ Some validations failed").red().bold());
                println!("   Failed: {}/{}", summary.failed, summary.total);
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("\n{} Error running tests: {}", style("⚠️").yellow(), e);
            std::process::exit(1);
        }
    }
}

fn count_neuroml_tests() -> Result<usize> {
    let test_file = "crates/pillars/src/api/neuroml/tests.rs";
    let content = std::fs::read_to_string(test_file)?;
    let count = content.lines().filter(|line| line.contains("#[test]")).count();
    Ok(count)
}

fn run_tests() -> Result<TestSummary> {
    // Run cargo test and capture output
    let output = Command::new("cargo")
        .args(["test", "-p", "pillars", "--lib", "neuroml", "--"])
        .output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Parse test results from cargo output
    let mut passed = 0;
    let mut failed = 0;
    let mut ignored = 0;
    
    // Look for lines like: test api::neuroml::tests::single_rs_neuron_spikes_under_pulse ... ok
    for line in stdout.lines().chain(stderr.lines()) {
        if line.contains("test result:") {
            // Parse: "test result: ok. 120 passed; 0 failed; 1 ignored"
            if let Some(metrics) = parse_test_result(line) {
                return Ok(metrics);
            }
        }
        
        if line.contains("passed") || line.contains("failed") || line.contains("ignored") {
            // Alternative format
            if line.contains("passed") {
                if let Some(n) = extract_number(line) {
                    passed = n;
                }
            }
            if line.contains("failed") {
                if let Some(n) = extract_number(line) {
                    failed = n;
                }
            }
            if line.contains("ignored") {
                if let Some(n) = extract_number(line) {
                    ignored = n;
                }
            }
        }
    }
    
    Ok(TestSummary {
        passed,
        failed,
        ignored,
        total: passed + failed,
    })
}

fn parse_test_result(line: &str) -> Option<TestSummary> {
    // Format: "test result: ok. 120 passed; 0 failed; 1 ignored; 0 measured"
    let parts: Vec<&str> = line.split(';').collect();
    
    let mut summary = TestSummary::default();
    
    for part in parts {
        let part = part.trim();
        
        if part.starts_with("ok.") {
            let nums: Vec<&str> = part.split_whitespace().collect();
            for (i, word) in nums.iter().enumerate() {
                if *word == "passed" && i > 0 {
                    if let Ok(n) = nums[i - 1].parse::<usize>() {
                        summary.passed = n;
                    }
                }
                if *word == "failed" && i > 0 {
                    if let Ok(n) = nums[i - 1].parse::<usize>() {
                        summary.failed = n;
                    }
                }
                if *word == "ignored" && i > 0 {
                    if let Ok(n) = nums[i - 1].parse::<usize>() {
                        summary.ignored = n;
                    }
                }
            }
        }
    }
    
    if summary.passed > 0 || summary.failed > 0 {
        summary.total = summary.passed + summary.failed;
        Some(summary)
    } else {
        None
    }
}

fn extract_number(line: &str) -> Option<usize> {
    line.split_whitespace()
        .filter_map(|word| word.parse::<usize>().ok())
        .next()
}

fn print_summary(summary: &TestSummary, verbose: bool) {
    println!("\n{}", style("📊 Validation Summary").bold());
    println!("{}", style("-".repeat(60)));
    
    println!("  Passed:  {}", style(format!("{:>4}", summary.passed)).green());
    println!("  Failed:  {}", style(format!("{:>4}", summary.failed)).red());
    println!("  Ignored: {}", style(format!("{:>4}", summary.ignored)).yellow());
    println!("  Total:   {}", style(format!("{:>4}", summary.total)).blue());
    println!();
    
    if verbose {
        println!("  Detailed output above");
    }
}

#[derive(Debug, Default)]
struct TestSummary {
    passed: usize,
    failed: usize,
    ignored: usize,
    total: usize,
}
