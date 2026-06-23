# 🧪 Research Modules — Cutting-Edge SNN + T3

Analysis of the coolest recent papers + implemented modules.

---

## 📚 Paper Analysis

### 1. 🧠 E-Prop (Eligibility Propagation) — Online SNN Learning

**Paper:** *"A solution to the learning dilemma for recurrent networks of spiking neurons"*
Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass (Nature Comms 2020)

**Key Insight:**
e-prop decomposes the learning gradient into **three local factors**, avoiding expensive Backprop-Through-Time:
1. **Eligibility trace** ($e_{ji}$) — local memory of pre/post spike correlation
2. **Learning signal** ($L_j$) — broadcast error from output
3. **Weight update**: $\Delta w_{ji} = \eta \cdot L_j(t) \cdot e_{ji}(t)$

The eligibility trace for a LIF neuron:
$$\frac{de_{ji}}{dt} = -\frac{e_{ji}}{\tau_e} + \psi_j(t) \cdot z_i(t)$$

Where $\psi_j(t) = \frac{\partial U_j}{\partial w_{ji}}$ (surrogate gradient) and $z_i(t)$ is pre-synaptic input.

**Why it matters for this engine:**
- Online learning without BPTT = perfect for streaming audio
- Local + broadcast = efficient CUDA implementation
- Complements existing STDP with supervised signal

**Implementation:** `src/neuron.rs` — `EPropLIF` struct + `eprop_train()` function
**Python wrapper:** `research/eprop_snn.py`

---

### 2. 💧 Liquid Time-Constant (LTC) Neurons

**Paper:** *"Liquid Time-Constant Networks"* — Hasani et al. (AAAI 2021 / Nature Mach. Intel. 2022)

**Key Insight:**
Standard LIF has a **fixed** time constant $\tau$. LTC makes $\tau$ **input-dependent**:
$$\tau(I(t)) = \text{softplus}(w_\tau \cdot I(t) + b_\tau)$$

This allows the neuron to dynamically adapt its integration window — speed up for fast transients, slow down for sustained signals.

**Discretized update:**
$$x_{t+1} = x_t + \Delta t \cdot \left( -\frac{x_t}{\tau(I_t)} + f(I_t, x_t) \right)$$

**Why it matters:**
- Fewer neurons needed for complex temporal tasks
- Naturally handles irregularly-sampled time series
- ODE-based = stable gradients, no exploding/vanishing

**Implementation:** `research/liquid_state_machine.py` uses LTC-like adaptation via theta modulation

---

### 3. 🔄 Spiking Reservoir Computing (Liquid State Machine)

**Papers:**
- *"Reservoir based spiking models for univariate Time Series"* (Frontiers 2023)
- *"Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations"* — Maass (2002)

**Key Insight:**
A **fixed**, randomly-connected spiking reservoir projects input into high-dimensional space. Only a simple **linear readout** is trained (e.g., NLMS, Ridge Regression).

**Why it matters:**
- **No backprop** through the reservoir — extremely fast to train
- Reservoir weights are static → can be heavily optimized (CUDA/ASM)
- NLMS readout already exists in this engine!
- Perfect partner for T3 oscillators as feature extractors

**Pipeline:**
```
Signal → Cochlea/Bandpass → T3 Oscillators → Spike Encoding → LIF Reservoir → NLMS Readout → Output
```

**Implementation:** `research/liquid_state_machine.py`

---

### 4. ⚡ Spiking Transformers (Spikformer)

**Paper:** *"Spikformer: When Spiking Neural Network Meets Transformer"* (Zhou et al., 2023)

**Key Insight:**
Replaces softmax attention with **Spiking Self-Attention (SSA)** — addition-only, no multiplications:
$$\text{SSA}(Q, K, V) = \text{Spike}(Q \cdot K^T) \cdot V$$

**Why it matters:**
- Addition-only = perfect for ASM/CUDA (like existing XOR attention)
- 0 float multiplications in attention
- Complements existing XOR attention (which uses popcnt)

**Status:** Not yet implemented — needs transformer infrastructure first.

---

### 5. 🎯 Surrogate Gradient Learning

**Papers:** *"Surrogate Gradient Learning in Spiking Neural Networks"* — Zenke & Neftci (2021)

**Key Insight:**
Replace the non-differentiable spike threshold with a smooth surrogate during backward pass:
$$\sigma'(v) \approx \frac{1}{\tau} \cdot \exp\left(-\frac{|v - v_{th}|}{\tau}\right)$$

**Implementation:** Used inside `EPropLIF` in `src/neuron.rs` as the $\psi_j(t)$ factor.

---

## 📦 Module Overview

| Module | File | What it does |
|:-------|:-----|:-------------|
| **Liquid State Machine** | `liquid_state_machine.py` | Spiking reservoir + NLMS readout |
| **E-Prop SNN** | `eprop_snn.py` | Online SNN learning with eligibility traces |
| **Paper analysis** | `README.md` | This file |

---

## 🚀 Quick Start

```bash
# Test LSM with T3 features
python research/liquid_state_machine.py

# Train SNN with E-Prop
python research/eprop_snn.py
```
