"""Text specs for connectome generation.

Each spec is a short English sentence describing a target brain. Categories
give semantically-adjacent groups so the trainer can report per-category
behaviour (and we can verify the generator separates them).
"""

SPECS: dict[str, list[str]] = {
    "motion_detection": [
        "High-speed visual T4/T5 motion detection connectome for drone flight stabilization with GABA inhibition",
        "Looming-sensitive visual circuit for high-speed collision avoidance in a flapping-wing robot",
        "Sparse temporal-difference motion estimation network for a bionic compound eye",
        "Direction-selective ganglion-cell circuit for optical flow computation in autonomous navigation",
        "Hyperacute motion detection connectome for F1 racing line following at 300 km/h",
    ],
    "olfactory_memory": [
        "Mushroom Body Kenyon Cell gamma associative olfactory memory circuit with dopaminergic MBON output",
        "Olfactory pattern-separation network for odor mixture discrimination via sparse coding",
        "Mushroom-body output neuron circuit converting odor identity into valence signals for foraging",
        "Honeybee olfactory short-term to long-term memory consolidation circuit with octopamine gating",
        "Antennal-lobe glomerular map for combinatorial odor encoding with lateral inhibition",
    ],
    "sonar_echo": [
        "Deep-sea submarine sonar hydroacoustic signal classification connectome with Hodgkin-Huxley biophysics",
        "Echolocation delay-line circuit for bat sonar ranging with coincidence detection",
        "Passive acoustic localization network for submarine hull-mounted hydrophone arrays",
        "Matched-filter echo-location classifier for underwater object recognition",
        "Multi-channel sonar beamforming network with adaptive notch filtering",
    ],
    "motor_gait": [
        "Central pattern generator for quadruped robot gallop locomotion with phase modulation",
        "Tripod-gait CPG circuit for hexapod walking over rough terrain with sensory feedback",
        "Swimming CPG for a bio-inspired robotic fish with undulatory propulsion",
        "Humanoid bipedal balance control network with vestibular-fused postural correction",
        "Brachiating CPG for a swinging primate-like robot arm",
    ],
    "reward_learning": [
        "Dopamine-gated reinforcement learning circuit for reward prediction error in an agent",
        "Basal-ganglia action-selection network with risk-sensitive tonic dopamine",
        "Opponent-process reward circuit for approach-avoidance decision making",
        "Neural architecture for temporal credit assignment in delayed-reward foraging",
        "Synaptic plasticity rule for maximizing entropy in curiosity-driven exploration",
    ],
    "visual_recognition": [
        "Scale-invariant object recognition hierarchy for an autonomous rover vision stack",
        "Face-selective fusiform-like network for robotic social interaction",
        "View-invariant 3D object recognition circuit from a convolutional preprocessor",
        "Low-light monocular depth estimation network for night-time drone navigation",
        "Attention-modulated visual search circuit with top-down feature feedback",
    ],
    "auditory": [
        "Pitch-shift-invariant auditory cortex model for cochlear implant processing",
        "Binaural sound-localization circuit using ITD/ILD coincidence maps",
        "Stream-segregation network for cocktail-party auditory scene analysis",
        "Spectral-temporal receptive field network for bird-song recognition",
        "Amplitude-modulation envelope detector circuit for speech prosody",
    ],
    "homeostasis": [
        "Homeostatic synaptic scaling circuit that maintains network firing rates",
        "Neural temperature-regulation connectome for thermoregulation in a robot",
        "Intrinsic-plasticity control loop for sensitivity adaptation",
        "Calcium-mediated homeostatic plasticity network for stable recurrent activity",
        "Sleep-wake oscillator circuit with neuromodulatory gating",
    ],
}

ALL_CATEGORIES = list(SPECS.keys())

def all_specs() -> list[tuple[str, str]]:
    """Flat list of (category, spec) pairs, in stable order."""
    out = []
    for cat in ALL_CATEGORIES:
        for s in SPECS[cat]:
            out.append((cat, s))
    return out
