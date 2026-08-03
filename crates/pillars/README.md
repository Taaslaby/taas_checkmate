# 🧠 Pillars — Zero-Cost SIMD Neural Dynamics

High-performance neural simulation engine with SIMD-first architecture and type-safe model construction.

## Features

- **Zero-cost abstractions**: SIMD operations compile to single CPU instructions
- **Type-safe builders**: No magic parameter indices, named configuration
- **Pre-configured patterns**: Izhikevich RS/FS/IB/CH, LIF, AdEx, HH, and more
- **Batch processing**: Process 8 neurons per instruction (AVX2)
- **Flexible architecture**: From single neurons to large-scale networks

## Quick Start

### Using Builder API (Recommended)

```rust
use pillars::prelude::*;

// Create 1000 Regular Spiking neurons
let mut pop = IzhikevichBuilder::new(1000)
    .dt(0.5)
    .pattern(IzhPattern::RegularSpiking)
    .build();

// Simulate
let input = vec![10.0; 1000];
let mut spikes = vec![0i32; 1000];
pop.step(&input, &mut spikes);
```

### Available Builders

#### Izhikevich Neurons

```rust
// Pre-configured patterns
let pop = IzhikevichBuilder::new(1000)
    .pattern(IzhPattern::RegularSpiking)      // RS, FS, IB, CH, LTS, TC, RZ
    .dt(0.5)
    .v_thresh(30.0)
    .build();

// Custom parameters
let pop = IzhikevichBuilder::new(1000)
    .a(0.02)
    .b(0.2)
    .c(-65.0)
    .d(8.0)
    .build();
```

#### LIF Neurons

```rust
let pop = LIFBuilder::new(2000)
    .dt(0.1)
    .tau_m(10.0)        // Membrane time constant (ms)
    .v_thresh(-54.0)    // Spike threshold (mV)
    .v_rest(-70.0)      // Resting potential (mV)
    .v_reset(-70.0)     // Reset potential (mV)
    .tau_ref(2.0)       // Refractory period (ms)
    .build();
```

#### AdEx Neurons

```rust
let pop = AdExBuilder::new(1000)
    .dt(0.05)
    .tau_m(5.0)         // Membrane time constant (ms)
    .tau_w(100.0)       // Adaptation time constant (ms)
    .delta_t(2.0)       // Exponential slope factor (mV)
    .a(0.5)             // Subthreshold adaptation (nS)
    .b(7.0)             // Spike-triggered adaptation (nA)
    .build();
```

### Manual Construction (Advanced)

```rust
use pillars::prelude::*;

let mut pop = NeuralPopulation::<IzhikevichModel>::new(1000);
pop.fill_param(0, 0.5);   // dt
pop.fill_param(1, 0.02);  // a
pop.fill_param(2, 0.2);   // b
// ... (not recommended for new code)
```

## Performance

- **8x throughput**: Process 8 neurons per SIMD instruction
- **FMA optimization**: Fused multiply-add for critical paths
- **Cache-friendly**: Column-major SoA layout
- **Zero allocation**: Pre-allocated aligned buffers

### Benchmark Results

```
LIF (1M neurons):     ~2.5 ms/step  (AVX2)
Izhikevich (1M):      ~3.8 ms/step  (AVX2)
AdEx (1M):            ~4.2 ms/step  (AVX2)
Hodgkin-Huxley (1M):  ~12 ms/step   (AVX2)
```

## Architecture

```
User Code
    ↓
api/Builder           ← Type-safe construction
    ↓
api/NeuralPopulation  ← Safe wrapper
    ↓
zoo/Models            ← Biological models (LIF, HH, STDP)
    ↓
ops/Primitives        ← ohmic, vtrap, sigmoid
    ↓
core/SimdLane         ← Generic SIMD abstraction
    ↓
Hardware (AVX2/NEON)
```

## Examples

Run examples with:

```bash
# Builder API demo
cargo run --example builder_demo

# Compare all Izhikevich patterns
cargo run --example izhikevich_patterns

# Large-scale network
cargo run --example network_demo --features rayon
```

## Documentation

- [API Reference](docs/PILLARS_API.md)
- [Performance Patterns](docs/PERFORMANCE_PATTERNS.md) *(coming soon)*
- [Mathematical Verification](docs/MATHEMATICAL_AUDIT.md)

## Supported Models

### Point Neurons
- **LIF**: Leaky Integrate-and-Fire
- **Izhikevich**: 7 pre-configured patterns (RS, FS, IB, CH, LTS, TC, RZ)
- **AdEx**: Adaptive Exponential I&F
- **FHN**: FitzHugh-Nagumo
- **HH**: Hodgkin-Huxley (Na⁺/K⁺ channels)
- **Morris-Lecar**: Ca²⁺/K⁺ channels
- **GLIF3/4**: Generalized LIF with adaptation

### Synapses
- **Exp2Syn**: Double-exponential conductance
- **NMDA**: Mg²⁺ block + voltage dependence
- **Tsodyks-Markram**: Short-term plasticity

### Plasticity
- **STDP**: Spike-timing dependent plasticity
- **BCM**: Bienenstock-Cooper-Munro
- **Homeostatic**: Synaptic scaling

## Features Flags

```toml
[dependencies]
pillars = { version = "0.1", features = ["rayon"] }
```

- `rayon`: Parallel population processing
- `fast-exp`: Polynomial exp approximation (~3x faster, ~1e-5 error)
- `t3`: T3 phase oscillator networks
- `train`: Reinforcement learning primitives
- `cuda`: GPU acceleration *(experimental)*

## License

MIT OR Apache-2.0

## Citation

If you use Pillars in research, please cite:

```bibtex
@software{pillars2024,
  title = {Pillars: Zero-Cost SIMD Neural Dynamics},
  author = {TaaS Team},
  year = {2024},
  url = {https://github.com/yourusername/taas_checkmate}
}
```