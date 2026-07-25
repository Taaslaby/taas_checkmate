import numpy as np

def test_somato_dendritic_python_api():
    import taas_checkmate as tc
    SomatoDendriticNeuron = tc.neurons.SomatoDendriticNeuron

    # Create population of 16 somato-dendritic neurons
    pop = SomatoDendriticNeuron(
        n_neurons=16,
        n_compartments=4,
        r_a=200.0,
        r_m=100.0,
        c_m=0.1,
        dt=0.5
    )

    # Enable active HH + Ca2+ channels
    pop.enable_active_channels(g_na=10.0, g_kv=5.0, g_ca=2.0)

    # Apply AMPA jump to distal compartment
    pop.jump_ampa(nrn=0, comp=3, g=0.10)

    # Run 50 simulation steps
    I_ext = np.zeros(16, dtype=np.float32)
    for _ in range(50):
        pop.step(I_ext, 0.5)

    v = pop.v
    spikes = pop.spikes
    v_dend = pop.v_dend(0, 3)

    print("✅ SomatoDendriticNeuron Python test successful!")
    print(f"Soma V[0]: {v[0]:.2f} mV, Distal Dendrite V[0, 3]: {v_dend:.2f} mV, Spikes: {spikes.sum()}")

if __name__ == "__main__":
    test_somato_dendritic_python_api()
