//! # Builder API Demo — Type-Safe Model Construction
//!
//! Demonstrates the new builder pattern for creating neuron populations
//! without magic parameter indices.
//!
//! Run with:
//! ```bash
//! cargo run --example builder_demo --features rayon
//! ```

use pillars::prelude::*;

fn main() {
    println!("🧠 Pillars Builder API Demo\n");

    // ═══════════════════════════════════════════════════════════════
    //  Example 1: Izhikevich Regular Spiking
    // ═══════════════════════════════════════════════════════════════
    
    println!("1️⃣  Izhikevich Regular Spiking (RS)");
    
    let mut pop_rs = IzhikevichBuilder::new(1000)
        .dt(0.5)
        .pattern(IzhPattern::RegularSpiking)
        .build();
    
    // Simulate with constant input
    let input = vec![10.0; 1000];
    let mut spikes = vec![0i32; 1000];
    
    for step in 0..100 {
        pop_rs.step(&input, &mut spikes);
        let spike_count: i32 = spikes.iter().sum();
        if spike_count > 0 {
            println!("   Step {}: {} spikes", step, spike_count);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    //  Example 2: Izhikevich Fast Spiking
    // ═══════════════════════════════════════════════════════════════
    
    println!("\n2️⃣  Izhikevich Fast Spiking (FS)");
    
    let mut pop_fs = IzhikevichBuilder::new(500)
        .dt(0.5)
        .pattern(IzhPattern::FastSpiking)
        .v_thresh(25.0)  // Custom threshold
        .build();
    
    let input = vec![15.0; 500];
    let mut spikes = vec![0i32; 500];
    
    for step in 0..50 {
        pop_fs.step(&input, &mut spikes);
        let spike_count: i32 = spikes.iter().sum();
        if spike_count > 0 {
            println!("   Step {}: {} spikes", step, spike_count);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    //  Example 3: LIF with Custom Parameters
    // ═══════════════════════════════════════════════════════════════
    
    println!("\n3️⃣  LIF with Custom Parameters");
    
    let mut pop_lif = LIFBuilder::new(2000)
        .dt(0.1)
        .tau_m(15.0)      // Slower membrane time constant
        .v_thresh(-50.0)  // Lower threshold
        .v_rest(-70.0)
        .v_reset(-75.0)   // Deeper reset
        .tau_ref(3.0)     // Longer refractory
        .build();
    
    let input = vec![25.0; 2000];
    let mut spikes = vec![0i32; 2000];
    
    for step in 0..100 {
        pop_lif.step(&input, &mut spikes);
        let spike_count: i32 = spikes.iter().sum();
        if step % 10 == 0 {
            println!("   Step {}: {} spikes", step, spike_count);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    //  Example 4: AdEx with Adaptation
    // ═══════════════════════════════════════════════════════════════
    
    println!("\n4️⃣  AdEx with Strong Adaptation");
    
    let mut pop_adex = AdExBuilder::new(1000)
        .dt(0.05)
        .tau_m(5.0)
        .tau_w(100.0)
        .delta_t(2.0)
        .a(2.0)           // Strong subthreshold adaptation
        .b(10.0)          // Strong spike-triggered adaptation
        .build();
    
    let input = vec![200.0; 1000];
    let mut spikes = vec![0i32; 1000];
    
    for step in 0..200 {
        pop_adex.step(&input, &mut spikes);
        let spike_count: i32 = spikes.iter().sum();
        if step % 20 == 0 {
            println!("   Step {}: {} spikes (adaptation kicks in)", step, spike_count);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════
    //  Example 5: Comparing Patterns
    // ═══════════════════════════════════════════════════════════════
    
    println!("\n5️⃣  Comparing All Izhikevich Patterns");
    
    let patterns = [
        ("RS", IzhPattern::RegularSpiking),
        ("IB", IzhPattern::IntrinsicallyBursting),
        ("CH", IzhPattern::Chattering),
        ("FS", IzhPattern::FastSpiking),
        ("LTS", IzhPattern::LowThresholdSpiking),
        ("TC", IzhPattern::ThalamoCortical),
        ("RZ", IzhPattern::Resonator),
    ];
    
    for (name, pattern) in patterns {
        let mut pop = IzhikevichBuilder::new(100)
            .dt(0.5)
            .pattern(pattern)
            .build();
        
        let input = vec![10.0; 100];
        let mut spikes = vec![0i32; 100];
        let mut total_spikes = 0;
        
        for _ in 0..200 {
            pop.step(&input, &mut spikes);
            total_spikes += spikes.iter().sum::<i32>();
        }
        
        println!("   {}: {} total spikes over 200 steps", name, total_spikes);
    }
    
    println!("\n✅ Builder API Demo Complete!");
    println!("\n💡 Key Benefits:");
    println!("   • No magic parameter indices");
    println!("   • Type-safe configuration");
    println!("   • Named parameters (readable code)");
    println!("   • Pre-configured patterns");
    println!("   • Fluent API (chainable methods)");
}
