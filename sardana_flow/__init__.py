"""sardana_flow — closed-loop connectome generation.

Text spec -> learned edge policy -> ConnectomeBrain (Rust) -> dynamics ->
reward -> REINFORCE. The Rust engine is the teacher: a generated connectome
is only as good as its behaviour when actually simulated.

Submodules:
    specs      — text specs (categories -> sentences)
    generator  — the policy: text -> latent positions -> edges (Gumbel/Bernoulli)
    metrics    — network-science scoring (modularity, clustering, heavy-tail...)
    env        — ConnectomeBrain wrapper + reward computation
    trainer    — REINFORCE-with-baseline loop
"""

__version__ = "0.1.0"
