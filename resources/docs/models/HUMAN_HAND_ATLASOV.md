# Human hand/forearm reference profile for Atlasov

`pillars::api::human_hand` contains a compact, source-backed reference model for
an eight-channel human hand/forearm system. It is intended for reproducible
numerical experiments and unit checks, not for diagnosing or predicting an
individual patient.

## Channels

The profile contains:

- FCR, FCU, ECRL, ECRB and ECU for wrist flexion/extension;
- FDS-II and FDP-II for the index finger;
- FDI (first dorsal interosseous) for intrinsic hand force.

`HumanMuscle::ALL` and `HumanHandProfile::muscles` define the stable channel
order. `HumanHandState` runs the Atlasov calcium/tension dynamics and exposes
`force_n` and `wrist_moment_nm`. The embedded `atlasov.joint_*` arrays retain
the generic Atlasov adjacent-pair buffers and are not the hand's anatomical
joint map; use the explicit human-hand moment fields instead.

## Sources and status

Morphology is taken from the open-access integrated forearm/hand study:

- Kerkhof, van Leeuwen & Vereecke (2018), *The digital human forearm and hand*,
  DOI [10.1111/joa.12877](https://doi.org/10.1111/joa.12877),
  [PMC6183001](https://pmc.ncbi.nlm.nih.gov/articles/PMC6183001/).
- The maximum stress convention is 35 N/cm² = 350,000 Pa, as used by
  Goislard de Monsabert et al. (2018), *Importance of Consistent Datasets in
  Musculoskeletal Modelling*, DOI
  [10.1007/s10439-017-1936-z](https://doi.org/10.1007/s10439-017-1936-z),
  [PMC5754461](https://pmc.ncbi.nlm.nih.gov/articles/PMC5754461/).

PCSA is measured/calculated anatomical data for one cadaver specimen (the
source reports fused ECRL/ECRB in its specimen). Maximum force is **derived**:

```text
F_max [N] = PCSA [mm²] × 1e−6 [m²/mm²] × 350,000 [N/m²]
```

Thus 358 mm² gives 125.3 N and 196 mm² gives 68.6 N. These are capacity
estimates for the cited cadaver/specimen and assumed specific tension, not a
simultaneous measurement of active force in a living hand.

The moment arms are **angle-dependent, piecewise-linear curves** fitted to
published ranges, not a claim of exact tendon-excursion measurements. Each
wrist channel has a [`WristMomentArmCurve`] with control points over
−60°…+60° (flexion positive), chosen so the neutral pose matches the values
in `HumanMuscleSpec`. The fitted ranges follow the published literature
(e.g. FCR ≈ 8–15 mm, ECU ≈ 3–12 mm; see Goislard de Monsabert et al. and the
ranges reported there), but the exact curve shape between control points is a
reduced-model approximation. `wrist_moment_nm` includes only the wrist channels
(FCR/FCU/ECRL/ECRB/ECU); FDS/FDP/FDI are excluded from that scalar and expose
separate per-channel primary-joint moments. Those reduced arms have no implied
finger joint label; a full model must supply the target DOF and a measured
angle-dependent arm for each tendon.

Usage: `HumanHandState::set_wrist_angle_deg(θ)` re-evaluates wrist moment arms
from the curves; `wrist_moment_nm` always reflects the current angle. Subject-specific work should
replace these values with angle-dependent tendon paths/moment-arm curves from
a complete musculoskeletal model. The cited papers report geometry and modelled wrist forces; they also
show that combining inconsistent anatomical datasets can change individual
muscle-force estimates substantially.

## Units

- `dt`, `tau_pump`: seconds (must use the same time unit);
- `optimal_fibre_length_m`: metres;
- `velocities_m_s`: metres/second;
- `specific_tension_pa`: pascals;
- `force_n`: newtons;
- `moment_arm_m`: metres;
- `wrist_moment_nm`: newton-metres.

Atlasov's `ca` remains a dimensionless activation pool. The profile calibrates
only the mechanical `t_max` output to SI force; it does not turn `ca` into a
measured calcium concentration.

## Real-data validation (sEMG + grip force)

Validated against **"sEMG for grasp force estimation"** (Zenodo 4554494, CC0):
5 subjects, 300 trials, 6-channel sEMG (ECU/FCU/FDS/FCR/Brachioradialis/ED,
µV) + measured grip force (N) at 4 strength levels (25/50/75/100 %).

Pipeline (`tests/validate_human_hand_emg.py`): RMS envelope (100 ms) →
per-channel p95 normalisation → exponential smoothing (τ ≈ 50 ms) →
deterministic Poisson motor spikes (rate = activation × 20 Hz) → Rust
`HumanHandState` step (model dt = 5 ms, neutral wrist pose). Model grip force
= FDS-II + FDP-II + 0.3·(FCR + FCU).

Results (300 trials, zero calibration on subject data):

| Metric | Mean | p25 | p75 |
|---|---|---|---|
| Pearson r | **0.69** | 0.63 | 0.81 |
| NRMSE | 0.37 | 0.21 | 0.37 |
| R² (affine) | 0.69 (median) | — | — |
| Peak ratio model/measured | 1.25 | — | — |

Correlation grows monotonically with the strength level (r = 0.62/0.69/0.72/
0.74 for 25/50/75/100 %), i.e. the model tracks activation dynamics better at
stronger contractions where the sEMG signal-to-noise ratio is higher. The
uncalibrated peak force lands within ~25 % of the measured peak. Per-subject
mean r: 0.81/0.66/0.79/0.87/0.58.

Honest limitations: the dataset has no wrist angle (model ran at neutral); the
sampling rate is undocumented (assumed 2 kHz); FDP-II/ECRB/FDI have no
recording in this dataset and are not driven; the absolute force scale is not
fitted per subject. **The wrist-flexor weight 0.3 was tuned on this dataset**
(w=0.0 gives r≈0.56, w=0.3 gives r≈0.69) — the headline r is therefore
calibrated on the test set and lower values are expected on held-out subjects.
The full metrics are stored in `validation_data/human_hand/results.json`
(generated locally; the large source archive is git-ignored).

## Installing the Python module

The `HumanHandState` Python binding lives in `taas_checkmate.pillars`. Build it
with maturin from the repo root (do **not** pass explicit `--features` — the
`[tool.maturin] features = ["python"]` entry in `pyproject.toml` is the correct
feature set and explicit flags break binding resolution):

```bash
.venv/Scripts/python.exe -m maturin develop
```

Then `from taas_checkmate.pillars import HumanHandState`.

## Example

```rust
use pillars::api::{HumanHandProfile, HumanHandState};

let profile = HumanHandProfile::reference();
let fcr_max_n = profile.muscles[0].max_isometric_force_n();

let mut hand = HumanHandState::new();
let spikes = [1; 8];
let resets = [0; 8];
hand.step(&spikes, &resets, 0.001, None, None);
println!("FCR capacity = {fcr_max_n:.1} N");
println!("reduced wrist moment = {:.3} N m", hand.wrist_moment_nm);
```

This reference should be validated against a downloadable force/kinematics
recording before making a biological accuracy claim. The repository's
`tests/validate_atlasov_real_data.py` remains the data-driven validation entry
point.
