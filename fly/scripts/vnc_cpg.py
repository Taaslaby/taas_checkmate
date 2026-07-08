#!/usr/bin/env python3
"""
VNC CPG Brain — Biological Ring Oscillator from MANC connectome.

Загружает Cluster 1 VNC нейронов (5,681 нейронов, 461K синапсов),
запускает LIF симуляцию через Rust backend.

Архитектура:
  VNC Cluster 1 → LIF → EN (efferent) rates → actuator drive
                    ↑ DN (descending) injection from brain
                    ↑ AN (ascending) feedback from MuJoCo

Формат бинарного файла (/tmp/vnc_cluster1.bin):
  [n_neurons: u32] [n_synapses: u32]
  [row_ptr: i32 × (n+1)]
  [col_indices: i32 × n_syn]
  [values: f32 × n_syn]
  [nt_types: f32 × n]  (1.0=exc, -1.0=inh)
  [root_id: u64 × n]
  [type_str_len: u32] [type_str: utf8] × n
  [class_str_len: u32] [class_str: utf8] × n
"""

import sys
import os
import struct
import time
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJ = Path(__file__).parent.parent.parent
if str(PROJ / 'target' / 'release') not in sys.path:
    sys.path.insert(0, str(PROJ / 'target' / 'release'))
try:
    import taas_checkmate as tcm
except ImportError:
    print("⚠ taas_checkmate not found. Build: cd taas_checkmate && cargo build --features python")
    tcm = None


class VNCCPGBrain:
    """VNC CPG Brain — биологический ring oscillator на LIF.

    Загружает Cluster 1 из MANC, идентифицирует DN/IN/AN/EN нейроны,
    запускает LIF динамику через Rust lif_sim_cpu_recording.
    """

    def __init__(self, path="/tmp/vnc_cluster1.bin", dt_ms=0.5, base_rate=50.0):
        self.dt_ms = dt_ms
        self.path = path
        self.base_rate = base_rate
        self.n = 0
        self.n_syn = 0
        self._last_spike_buf = None
        self._last_n_steps = 0

        # CSR data
        self.row_ptr = None
        self.col_indices = None
        self.values = None
        self.nt_types = None

        # Neuron metadata
        self.root_ids = []
        self.type_map = {}     # idx → type string
        self.class_map = {}    # idx → class string

        # Neuron group indices
        self.dn_indices = []   # descending (input from brain)
        self.an_indices = []   # ascending (feedback from legs)
        self.en_indices = []   # efferent (output to actuators)
        self.in_indices = []   # intrinsic (ring oscillator core)

        self._load()

    def _load(self):
        """Load CSR binary exported by export_vnc_cluster1.py."""
        t0 = time.time()
        print(f"  [VNC.CPG] Loading {self.path}...")

        with open(self.path, 'rb') as f:
            data = f.read()

        offset = 0
        self.n = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        self.n_syn = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        # row_ptr
        self.row_ptr = np.frombuffer(data, dtype=np.int32, count=self.n + 1, offset=offset)
        offset += (self.n + 1) * 4

        # col_indices
        self.col_indices = np.frombuffer(data, dtype=np.int32, count=self.n_syn, offset=offset)
        offset += self.n_syn * 4

        # values
        self.values = np.frombuffer(data, dtype=np.float32, count=self.n_syn, offset=offset)
        offset += self.n_syn * 4

        # nt_types
        self.nt_types = np.frombuffer(data, dtype=np.float32, count=self.n, offset=offset)
        offset += self.n * 4

        # root_ids
        self.root_ids = list(np.frombuffer(data, dtype=np.uint64, count=self.n, offset=offset))
        offset += self.n * 8

        # Type/class strings
        for i in range(self.n):
            s_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            typ = data[offset:offset + s_len].decode('utf-8', errors='replace')
            offset += s_len
            c_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            cls = data[offset:offset + c_len].decode('utf-8', errors='replace')
            offset += c_len
            self.type_map[i] = typ
            self.class_map[i] = cls

        # Identify neuron groups
        for i in range(self.n):
            cls = self.class_map.get(i, '').lower()
            typ = self.type_map.get(i, '').lower()
            if 'descending' in cls or 'dn' in typ:
                self.dn_indices.append(i)
            if 'ascending' in cls or 'an' in typ:
                self.an_indices.append(i)
            if 'efferent' in cls or 'en' in typ:
                self.en_indices.append(i)
            if 'intrinsic' in cls or 'in' in typ:
                self.in_indices.append(i)

        self.n_padded = (self.n + 7) // 8 * 8

        # LIF state
        self.v = np.full(self.n_padded, -52.0, dtype=np.float32)
        self.v[:self.n] += np.random.RandomState(42).uniform(-1.0, 1.0, self.n).astype(np.float32)
        self.I_buf = np.zeros(self.n_padded, dtype=np.float32)
        self.spikes = np.zeros(self.n_padded, dtype=np.int32)

        # LIF params
        self.tau_m = np.full(self.n_padded, 20.0, dtype=np.float32)
        self.v_rest = np.full(self.n_padded, -52.0, dtype=np.float32)
        self.v_thresh = np.full(self.n_padded, -45.0, dtype=np.float32)
        self.v_reset = np.full(self.n_padded, -52.0, dtype=np.float32)

        # Alpha-synapse
        self.tau_syn = 5.0
        self.g_syn = np.zeros(self.n_padded, dtype=np.float32)
        self.delay_steps = 4
        self.delay_buf = np.zeros(self.n_padded * self.delay_steps, dtype=np.float32)
        self.delay_pos = np.zeros(1, dtype=np.int32)

        # External input (injection to DN + noise to IN)
        self.external_I = np.zeros(self.n_padded, dtype=np.float32)

        # Poisson noise
        self._poisson_rng = np.random.RandomState(42)
        self._base_rate = self.base_rate

        # Rust backend
        self.use_rust = (tcm is not None and
                         hasattr(tcm, 'lif_connectome') and
                         hasattr(tcm.lif_connectome, 'lif_sim_cpu_recording'))

        # Recording
        self._recorder = None
        if tcm is not None and hasattr(tcm, 'SpikeRecorder'):
            try:
                self._recorder = tcm.SpikeRecorder(self.n)
            except Exception:
                pass

        # EN rate cache
        self._en_rates = np.zeros(len(self.en_indices), dtype=np.float32)
        self._dn_injection = np.zeros(len(self.dn_indices), dtype=np.float32)

        print(f"  [VNC.CPG] Loaded: {self.n:,} neurons, {self.n_syn:,} synapses "
              f"[{time.time()-t0:.1f}s]")
        print(f"  [VNC.CPG] Groups: DN={len(self.dn_indices)} IN={len(self.in_indices)} "
              f"AN={len(self.an_indices)} EN={len(self.en_indices)}")
        print(f"  [VNC.CPG] Backend: {'Rust LIF' if self.use_rust else 'Python fallback'}")

    def inject_dn(self, dn_rates: np.ndarray):
        """Inject descending neuron rates from brain (0.0-1.0)."""
        self._dn_injection = np.array(dn_rates, dtype=np.float32)

    def get_en_rates(self) -> np.ndarray:
        """Get efferent neuron rates [0, 1] for actuator control."""
        return self._en_rates.copy()

    def get_an_rates(self) -> np.ndarray:
        """Get ascending neuron rates [0, 1] for sensory feedback."""
        if self._last_spike_buf is None or self._last_n_steps == 0:
            return np.zeros(len(self.an_indices), dtype=np.float32)
        an_spikes = np.zeros(len(self.an_indices), dtype=np.int32)
        n_steps = self._last_n_steps
        for step in range(n_steps):
            offset = step * self.n
            for i, an_idx in enumerate(self.an_indices):
                if an_idx < self.n and self._last_spike_buf[offset + an_idx]:
                    an_spikes[i] += 1
        batch_time_s = n_steps * self.dt_ms / 1000.0
        return np.clip(an_spikes.astype(np.float32) / (batch_time_s * 100.0), 0.0, 1.0)

    def step_batch(self, t_sim: float):
        """Run LIF batch on VNC cluster 1.

        Injects:
        - Poisson noise into IN neurons (ring oscillator drive)
        - DN rates from brain (descending modulation)

        Returns: EN rates [0, 1]
        """
        n_steps = max(1, int(25.0 / self.dt_ms))
        dt = self.dt_ms

        # ─── Set external input ───
        self.external_I.fill(0.0)

        # Continuous Gaussian noise for IN neurons (kicks the ring)
        # FlyWire uses ~150Hz Poisson → ~10 pA baseline. Here: Gaussian with σ=5 pA
        if len(self.in_indices) > 0:
            in_arr = np.array(self.in_indices, dtype=np.int32)
            noise = self._poisson_rng.randn(len(in_arr)).astype(np.float32) * 5.0
            self.external_I[in_arr] = noise + 8.0  # baseline 8 pA + noise

        # DN injection from brain — continuous current, not pulse
        for i, rate in enumerate(self._dn_injection):
            if i < len(self.dn_indices):
                idx = self.dn_indices[i]
                if rate > 0.01:
                    self.external_I[idx] += rate * 30.0  # continuous DN drive

        # ─── Run LIF ───
        self.I_buf.fill(0.0)
        self.I_buf[:self.n] = self.external_I[:self.n]

        spike_buf = np.zeros(n_steps * self.n, dtype=np.int32)

        if self.use_rust:
            try:
                tcm.lif_connectome.lif_sim_cpu_recording(
                    self.row_ptr, self.col_indices, self.values,
                    n_steps, dt,
                    self.v, self.tau_m, self.v_rest, self.v_thresh, self.v_reset,
                    self.nt_types, self.external_I, self.I_buf, self.spikes,
                    spike_buf,
                    self.g_syn, self.tau_syn,
                    self.delay_buf, self.delay_steps, self.delay_pos,
                )
            except Exception as e:
                print(f"  ⚠ VNC LIF error: {e}")
                self.use_rust = False
        else:
            # Python fallback
            v = self.v[:self.n].copy()
            for step in range(n_steps):
                alpha = dt / 10.0
                v = v * (1 - alpha) + self.I_buf[:self.n] * alpha + self.v_rest[0] * alpha
                spiked = v >= self.v_thresh[0]
                v[spiked] = self.v_reset[0]
                spike_buf[step * self.n:(step + 1) * self.n] = spiked.astype(np.int32)
                self.I_buf[:self.n] = self.external_I[:self.n]
                for pre_idx in np.where(spiked)[0]:
                    s, e = int(self.row_ptr[pre_idx]), int(self.row_ptr[pre_idx + 1])
                    for j in range(s, e):
                        post = int(self.col_indices[j])
                        if post < self.n:
                            self.I_buf[post] += self.values[j] * self.nt_types[pre_idx]
            self.v[:self.n] = v

        self._last_spike_buf = spike_buf
        self._last_n_steps = n_steps

        # ─── Extract EN rates ───
        if self._recorder is not None and len(self.en_indices) > 0:
            self._en_rates = np.array(
                self._recorder.record_group(spike_buf, n_steps, self.dt_ms,
                                            self.en_indices, 100.0),
                dtype=np.float32,
            )
        else:
            self._extract_en_rates_py(spike_buf, n_steps)

        return self._en_rates

    def _extract_en_rates_py(self, spike_buf, n_steps):
        """Python fallback for EN rate extraction."""
        en_spikes = np.zeros(len(self.en_indices), dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            for i, en_idx in enumerate(self.en_indices):
                if en_idx < self.n and spike_buf[offset + en_idx]:
                    en_spikes[i] += 1
        batch_time_s = n_steps * self.dt_ms / 1000.0
        self._en_rates = np.clip(
            en_spikes.astype(np.float32) / (batch_time_s * 100.0),
            0.0, 1.0,
        )

    def print_stats(self):
        """Print oscillation diagnostics."""
        if self._last_spike_buf is None or self._last_n_steps == 0:
            print("  [VNC.CPG] No data yet")
            return
        n_steps = self._last_n_steps
        spike_buf = self._last_spike_buf

        total_spikes = int(np.sum(spike_buf))
        dur = n_steps * self.dt_ms / 1000.0

        print(f"  [VNC.CPG] Stats ({dur:.2f}s real, {n_steps} steps):")
        print(f"    Total spikes: {total_spikes}")
        print(f"    Mean rate: {total_spikes / self.n / dur:.1f} Hz")

        # Per-group rates
        for name, indices in [('DN', self.dn_indices), ('IN', self.in_indices),
                               ('AN', self.an_indices), ('EN', self.en_indices)]:
            if not indices:
                continue
            group_spikes = 0
            for step in range(n_steps):
                offset = step * self.n
                for idx in indices:
                    if idx < self.n and spike_buf[offset + idx]:
                        group_spikes += 1
            rate = group_spikes / len(indices) / dur if len(indices) > 0 else 0
            bars = '█' * min(20, int(rate))
            print(f"    {name:4s}: {rate:6.1f} Hz  {bars}")


class VNCHHCPGBrain:
    """VNC CPG Brain using HH GPU dynamics on MANC Cluster 1 connectome.

    Replaces LIF with Hodgkin-Huxley for biological realism.
    Uses GpuHHLUT via Rust step_batch for performance.

    Architecture:
      VNC Cluster 1 → HH → EN (efferent) rates → actuator drive
                        ↑ DN (descending) injection from brain
                        ↑ AN (ascending) feedback from MuJoCo
    """

    def __init__(self, path="/tmp/vnc_cluster1.bin", dt_ms=0.01,
                 hh_dt=0.01, base_rate=50.0, pacemaking=False):
        self.dt_ms = dt_ms
        self.hh_dt = hh_dt
        self.path = path
        self.base_rate = base_rate
        self.n = 0
        self.n_syn = 0
        self._last_spike_buf = None
        self._last_n_steps = 0

        # CSR data
        self.row_ptr = None
        self.col_indices = None
        self.values = None
        self.nt_types = None
        self._prop_scale = 1.0

        # Neuron metadata
        self.root_ids = []
        self.type_map = {}
        self.class_map = {}

        # Neuron group indices
        self.dn_indices = []
        self.an_indices = []
        self.en_indices = []
        self.in_indices = []

        # HH state
        self.I_syn = None
        self.sim = None
        self._use_gpu = False
        self._batch_count = 0
        self._pacemaking = pacemaking

        self._load()

    def _load(self):
        """Load CSR binary and initialize HH GPU."""
        t0 = time.time()
        print(f"  [VNC.HH] Loading {self.path}...")

        with open(self.path, 'rb') as f:
            data = f.read()

        offset = 0
        self.n = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        self.n_syn = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        self.row_ptr = np.frombuffer(data, dtype=np.int32, count=self.n + 1, offset=offset)
        offset += (self.n + 1) * 4
        self.col_indices = np.frombuffer(data, dtype=np.int32, count=self.n_syn, offset=offset)
        offset += self.n_syn * 4
        self.values = np.frombuffer(data, dtype=np.float32, count=self.n_syn, offset=offset)
        offset += self.n_syn * 4
        self.nt_types = np.frombuffer(data, dtype=np.float32, count=self.n, offset=offset)
        offset += self.n * 4
        self.root_ids = list(np.frombuffer(data, dtype=np.uint64, count=self.n, offset=offset))
        offset += self.n * 8

        for i in range(self.n):
            s_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            typ = data[offset:offset + s_len].decode('utf-8', errors='replace')
            offset += s_len
            c_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            cls = data[offset:offset + c_len].decode('utf-8', errors='replace')
            offset += c_len
            self.type_map[i] = typ
            self.class_map[i] = cls

        for i in range(self.n):
            cls = self.class_map.get(i, '').lower()
            typ = self.type_map.get(i, '').lower()
            if 'descending' in cls or 'dn' in typ:
                self.dn_indices.append(i)
            if 'ascending' in cls or 'an' in typ:
                self.an_indices.append(i)
            if 'efferent' in cls or 'en' in typ:
                self.en_indices.append(i)
            if 'intrinsic' in cls or 'in' in typ:
                self.in_indices.append(i)

        # HH GPU setup
        self.I_syn = np.zeros(self.n, dtype=np.float32)
        self.external_I = np.zeros(self.n, dtype=np.float32)
        self.tau_syn = 5.0

        if tcm is not None and hasattr(tcm.hh, 'GpuHHLUT'):
            try:
                self.sim = tcm.hh.GpuHHLUT(self.n, self.hh_dt, pacemaking=self._pacemaking)
                self._use_gpu = True
                print(f"  [VNC.HH] GPU HH backend: {self.n} neurons")
            except Exception as e:
                print(f"  [VNC.HH] GPU init failed: {e}")

        self._recorder = None
        if tcm is not None and hasattr(tcm, 'SpikeRecorder'):
            try:
                self._recorder = tcm.SpikeRecorder(self.n)
            except Exception:
                pass

        self._en_rates = np.zeros(len(self.en_indices), dtype=np.float32)
        self._dn_injection = np.zeros(len(self.dn_indices), dtype=np.float32)
        self._poisson_rng = np.random.RandomState(42)

        print(f"  [VNC.HH] Loaded: {self.n:,} neurons, {self.n_syn:,} synapses "
              f"[{time.time()-t0:.1f}s]")
        print(f"  [VNC.HH] Groups: DN={len(self.dn_indices)} IN={len(self.in_indices)} "
              f"AN={len(self.an_indices)} EN={len(self.en_indices)}")
        print(f"  [VNC.HH] Backend: {'GPU HH' if self._use_gpu else 'no GPU'}")

    def inject_dn(self, dn_rates: np.ndarray):
        """Inject descending neuron rates from brain (0.0-1.0)."""
        self._dn_injection = np.array(dn_rates, dtype=np.float32)

    def get_en_rates(self) -> np.ndarray:
        """Get efferent neuron rates [0, 1] for actuator control."""
        return self._en_rates.copy()

    def get_an_rates(self) -> np.ndarray:
        """Get ascending neuron rates [0, 1] for sensory feedback."""
        if self._last_spike_buf is None or self._last_n_steps == 0:
            return np.zeros(len(self.an_indices), dtype=np.float32)
        an_spikes = np.zeros(len(self.an_indices), dtype=np.int32)
        n_steps = self._last_n_steps
        for step in range(n_steps):
            offset = step * self.n
            for i, an_idx in enumerate(self.an_indices):
                if an_idx < self.n and self._last_spike_buf[offset + an_idx]:
                    an_spikes[i] += 1
        batch_time_s = n_steps * self.dt_ms / 1000.0
        return np.clip(an_spikes.astype(np.float32) / (batch_time_s * 100.0), 0.0, 1.0)

    def step_batch(self, t_sim: float):
        """Run HH batch on VNC cluster 1 with GPU acceleration."""
        n_steps = max(1, int(25.0 / self.dt_ms))
        self._batch_count += 1

        self.external_I.fill(0.0)

        # Gaussian noise for IN neurons
        if len(self.in_indices) > 0:
            in_arr = np.array(self.in_indices, dtype=np.int32)
            noise = self._poisson_rng.randn(len(in_arr)).astype(np.float32) * 5.0
            self.external_I[in_arr] = noise + 8.0

        # DN injection from brain
        for i, rate in enumerate(self._dn_injection):
            if i < len(self.dn_indices):
                idx = self.dn_indices[i]
                if rate > 0.01:
                    self.external_I[idx] += rate * 30.0

        self.I_syn.fill(0.0)

        if self._use_gpu:
            spike_buf = self.sim.step_batch(
                self.external_I, self.I_syn, n_steps, self.tau_syn, self.hh_dt,
                self.row_ptr, self.col_indices, self.values,
                self.nt_types, self._prop_scale,
            )
        else:
            spike_buf = np.zeros(n_steps * self.n, dtype=np.int32)

        self._last_spike_buf = spike_buf
        self._last_n_steps = n_steps

        # Extract EN rates
        if self._recorder is not None and len(self.en_indices) > 0:
            self._en_rates = np.array(
                self._recorder.record_group(spike_buf, n_steps, self.dt_ms,
                                            self.en_indices, 100.0),
                dtype=np.float32,
            )
        else:
            self._extract_en_rates_py(spike_buf, n_steps)

        return self._en_rates

    def _extract_en_rates_py(self, spike_buf, n_steps):
        """Python fallback for EN rate extraction."""
        en_spikes = np.zeros(len(self.en_indices), dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            for i, en_idx in enumerate(self.en_indices):
                if en_idx < self.n and spike_buf[offset + en_idx]:
                    en_spikes[i] += 1
        batch_time_s = n_steps * self.dt_ms / 1000.0
        self._en_rates = np.clip(
            en_spikes.astype(np.float32) / (batch_time_s * 100.0),
            0.0, 1.0,
        )

    def print_stats(self):
        """Print oscillation diagnostics."""
        if self._last_spike_buf is None or self._last_n_steps == 0:
            print("  [VNC.HH] No data yet")
            return
        n_steps = self._last_n_steps
        spike_buf = self._last_spike_buf
        total_spikes = int(np.sum(spike_buf))
        dur = n_steps * self.dt_ms / 1000.0

        print(f"  [VNC.HH] Stats ({dur:.2f}s real, {n_steps} steps):")
        print(f"    Total spikes: {total_spikes}")
        print(f"    Mean rate: {total_spikes / self.n / dur:.1f} Hz")

        for name, indices in [('DN', self.dn_indices), ('IN', self.in_indices),
                               ('AN', self.an_indices), ('EN', self.en_indices)]:
            if not indices:
                continue
            group_spikes = 0
            for step in range(n_steps):
                offset = step * self.n
                for idx in indices:
                    if idx < self.n and spike_buf[offset + idx]:
                        group_spikes += 1
            rate = group_spikes / len(indices) / dur if len(indices) > 0 else 0
            bars = '█' * min(20, int(rate))
            print(f"    {name:4s}: {rate:6.1f} Hz  {bars}")
