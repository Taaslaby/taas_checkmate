# Cognitive Arena Web UI

This UI is a renderer for the Rust-owned `CognitiveArena` state machine. It does
not contain a second simulator.

## Build the Rust/WASM package

Install the target and `wasm-pack` once:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Then run the single reproducible build script (git-bash / any POSIX shell):

```bash
resources/web/cognitive-arena/build-wasm.sh
```

This is the only supported build path. (wasm-pack 0.15 rejects `--out-dir`
and positional manifest paths, and its `--artifact-dir` flag requires nightly
cargo, so the script builds in the repo root with `--no-default-features
--features wasm` and moves the generated `pkg/` into this directory.)

The generated `pkg/` directory is intentionally build output, is git-ignored,
and should not be hand-edited. `app.js` loads its generated
`taas_checkmate.js` glue module with a dynamic `import()` so that a missing
`pkg/` degrades to a visible offline state instead of a silent module failure.

## Run locally

A module-aware static server is required; opening `index.html` with `file://`
will be blocked by browser module/WASM security rules.

```bash
python -m http.server 8080 -d resources/web/cognitive-arena
```

Open <http://127.0.0.1:8080/>.

## Boundary contract

```text
CognitiveArenaWasm::new()
  ├─ snapshot() -> JSON snapshot, schema_version = 3
  ├─ step()    -> one Rust tick + JSON snapshot
  ├─ run(n)    -> n Rust ticks + JSON snapshot
  ├─ train(n, max_steps) -> n R-STDP training episodes + JSON snapshot
  └─ reset()   -> reset Rust state + JSON snapshot
```

The browser only calls these methods and renders the returned snapshot.
Mission 02 (Two-Key Vault) snapshot fields:

```text
red_key / blue_key / decoy / door   object positions
obstacles + per-agent known maps    partial observation
scout / carrier (role, position, action, carrying, partner_key_known,
                memory confidence/retention, decisions, path steps)
scout/carrier.brain: sensor_rates[8], workspace_rates[8],
                    action_rates[5], weight_heat[5×64] (W2 per action ×
                    workspace neuron), dopamine, reward, baseline,
                    spikes[sensor, workspace, output]
messages_sent, coordination_delay, decoy_hits
brain_activity (mean spikes/tick), training {episodes, rewards, steps}
belief_uncertainty, memory_retention, events
```

## Mission 03: Hide & Seek (emergent)

Two **hiders** vs three **seekers** on a large procedurally generated complex
(48×36, rooms/doorways/shelters/boxes). Each of the five agents has its own
spiking brain (pillars LIF + spiking GNW + R-STDP). No expert-pipeline
shaping: rewards are emergent — survival ticks, flee/chase distance
potentials, barricade/unblock events, tags.

```text
Hider:  SURVIVE_TICK + flee_potential(Δdistance-to-threat)
        + BARRICADE(push box into doorway that blocks paths)
        − TAGGED + ROUND_SURVIVED
Seeker: −TIME_PRESSURE + chase_potential(Δdistance-to-hider)
        + TAG + UNBLOCK
```

Boxes are movable: pushing one into a doorway barricades it (BFS paths
lengthen or vanish); seekers can push it back out. The learning curve in the
UI plots hider survival (steps) and tags per episode — because both teams
learn, the curve oscillates instead of saturating, which is the honest
signature of competitive emergence.

Boundary contract (same Rust-owned pattern):

```text
HideSeekWasm::new()
  ├─ snapshot() -> JSON snapshot, schema_version = 4, mission = "hide_seek"
  ├─ step() / run(n) / train(n, max_steps) / reset()
```

## The spiking brain

Each agent is controlled by a real spiking network built on the pillars LIF
kernel (`SpikingBrain`):

```text
perception (8 channels)  →  sensor population 8×8 LIF
                        →  GNW workspace 8×8 LIF (global broadcast)
                        →  output population 5×8 LIF → action
```

Learning is **reward-modulated STDP** (3-factor): eligibility traces track
workspace→output coincidences; dopamine = reward − baseline gates the weight
update; per-action weight normalization keeps the drive stable. `train()`
perturbs the prior weights (confused actions), runs episodes with shaping
reinforcing the expert mission pipeline, and the curve + W2 heatmaps in the UI
show the brains recovering and sharpening.

Semantics: each agent perceives only its own key color; key positions are
exchanged via semantic-pointer bind/unbind; the vault opens only when both
agents stand at the door each carrying their key; a decoy trap is learned into
the agent's obstacle map after the first hit; a long wait at the vault latches
a meet-up navigation toward the decoded partner key position (a one-shot
fallback — the default mission coordinates within 10 steps, so `mutual_aid`
appears only in long-delay edge cases, which are covered by synthetic tests).
