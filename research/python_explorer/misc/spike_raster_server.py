#!/usr/bin/env python3
"""
Live spike raster server.
Loads FlyWire 138K connectome, streams spike raster data via WebSocket.
"""

import sys, os, json, asyncio, argparse
import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(__file__))
from bench_connectome import load_connectome, setup_state, pad_csr

N_VISIBLE = 500
STEPS_PER_FRAME = 5
DT = 0.5
SEED = 42
HOST = "0.0.0.0"
PORT = 8765


def simulate_connectome(n, row_ptr, col_indices, weights, s, visible_idxs):
    """Generator: yields (visible_spikes_list, step, total_spikes)."""
    from taas_checkmate import step_izhikevich, propagate_spikes

    n_real = n
    step = 0

    while True:
        spiked_visible = []
        for _ in range(STEPS_PER_FRAME):
            prev_spiked = np.where(s["spikes"][:n_real] != 0)[0]
            s["I_buf"][:] = s["external_I"][:]
            if len(prev_spiked) > 0:
                propagate_spikes(
                    row_ptr, col_indices, weights,
                    prev_spiked.astype(np.int32),
                    s["nt_types"], s["I_buf"], 1.0,
                )
            step_izhikevich(
                s["v"], s["u"], s["trace"],
                s["a"], s["b"], s["I_buf"],
                DT, 0.99, s["spikes"], s["c"], s["d"],
            )
            step += 1

            # Map global neuron indices to visible indices
            v_binary = s["spikes"][visible_idxs] != 0
            v_list = np.where(v_binary)[0].tolist()
            spiked_visible.append(v_list)

        total_spikes = int(np.count_nonzero(s["spikes"][:n_real]))
        yield spiked_visible, step, total_spikes


def setup_spontaneous(n, s):
    rng = np.random.default_rng(SEED)
    s["external_I"][:] = rng.uniform(0, 5, s["n"])


connections = set()
peers = set()


async def handler(ws):
    connections.add(ws)
    peers.add(ws)
    try:
        n, row_ptr, col_indices, weights = load_connectome()
        s = setup_state(n)
        setup_spontaneous(n, s)

        rng = np.random.default_rng(SEED + 1)
        visible_idxs = rng.choice(n, N_VISIBLE, replace=False)
        visible_idxs.sort()

        row_ptr_p, col_idx_p, weights_p = pad_csr(
            row_ptr, col_indices, weights, s["n"]
        )

        await ws.send(json.dumps({
            "type": "init",
            "n_neurons": n,
            "n_visible": N_VISIBLE,
            "n_synapses": len(weights),
            "dt": DT,
            "steps_per_frame": STEPS_PER_FRAME,
        }))

        gen = simulate_connectome(n, row_ptr_p, col_idx_p,
                                   weights_p, s, visible_idxs)

        for spiked_list, step, total_spikes in gen:
            msg = json.dumps({
                "type": "frame",
                "spikes": spiked_list,
                "step": step,
                "total_spikes": total_spikes,
                "firing_rate": total_spikes / (step * DT * 0.001),
            })
            await ws.send(msg)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connections.discard(ws)
        peers.discard(ws)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--visible", type=int, default=N_VISIBLE)
    parser.add_argument("--steps-per-frame", type=int, default=STEPS_PER_FRAME)
    args = parser.parse_args()

    print(f"Spike raster server: ws://{HOST}:{args.port}")
    print(f"  Visible: {args.visible} neurons (random sample)")
    print(f"  Steps/frame: {args.steps_per_frame}")
    print(f"  DT: {DT}ms | Rate: {1000/(DT*args.steps_per_frame):.0f} Hz")
    print(f"  Connectome loads on first connection.")

    async with websockets.serve(handler, HOST, args.port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
