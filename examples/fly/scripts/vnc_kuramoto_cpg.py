#!/usr/bin/env python3
"""VNC Kuramoto CPG — Biological Ring Oscillator via Kuramoto dynamics.

Загружает VNC Cluster 1 (5,681 GABA нейронов), запускает Kuramoto динамику
на реальных весах коннектома. Реципрокное торможение → антифазные осцилляции.

Kuramoto: dθ/dt = ω + Σ W_ij · sin(θ_j - θ_i)
  W_ij < 0 для GABA (ингибиторные) → anti-phase
  W_ij > 0 для ACh/GLUT (эксайтаторные) → in-phase

Архитектура:
  Kuramoto Cluster 1 → sin(θ) → actuator drive (tripod A / tripod B)
                    ↑ DN (descending) modulation via omega shift
"""
import sys
import struct
import time
import math
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent.parent.parent
if str(PROJ / 'target' / 'release') not in sys.path:
    sys.path.insert(0, str(PROJ / 'target' / 'release'))
try:
    import taas_checkmate as tcm
except ImportError:
    tcm = None


class VNCKuramotoCPG:
    """VNC CPG using Kuramoto dynamics on MANC Cluster 1 connectome.

    Все 5,681 нейронов — GABA (ингибиторные), поэтому веса coupling инвертированы:
    Kuramoto W_ij = -raw_weight  (anti-phase drive).

    Выход:
    - get_tripod_rates() → (rate_A, rate_B) for tripod gait
    - get_phase(i) → raw phase of i-th neuron
    """

    def __init__(self, path="/tmp/vnc_cluster1.bin", dt_s=0.005, omega_base=8.0, coupling_scale=30.0):
        """
        Args:
            path: путь к бинарнику VNC Cluster 1 CSR
            dt_s: шаг Kuramoto (сек), 5ms = 200 Hz update
            omega_base: базовая частота осцилляций (рад/с), 8 рад/с ≈ 1.3 Hz
            coupling_scale: глобальный множитель coupling (30=сильный, 10=слабый)
        """
        self.dt_s = dt_s
        self.omega_base = omega_base
        self.coupling_scale = coupling_scale
        self._load(path)

    def _load(self, path):
        t0 = time.time()
        print(f"  [VNC.KUR] Loading {path}...")

        with open(path, 'rb') as f:
            data = f.read()

        offset = 0
        self.n = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        self.n_syn = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        # CSR arrays
        self.row_ptr = np.frombuffer(data, dtype=np.int32, count=self.n + 1, offset=offset).copy()
        offset += (self.n + 1) * 4
        self.col_indices = np.frombuffer(data, dtype=np.int32, count=self.n_syn, offset=offset).copy()
        offset += self.n_syn * 4
        self.weights = np.frombuffer(data, dtype=np.float32, count=self.n_syn, offset=offset).copy()
        offset += self.n_syn * 4
        self.nt_types = np.frombuffer(data, dtype=np.float32, count=self.n, offset=offset).copy()
        offset += self.n * 4
        self.root_ids = list(np.frombuffer(data, dtype=np.uint64, count=self.n, offset=offset))
        offset += self.n * 8

        # Type/class strings
        self.type_map = {}
        self.class_map = {}
        for i in range(self.n):
            s_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            self.type_map[i] = data[offset:offset + s_len].decode('utf-8', errors='replace')
            offset += s_len
            c_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            self.class_map[i] = data[offset:offset + c_len].decode('utf-8', errors='replace')
            offset += c_len

        # Identify neuron groups
        self.dn_indices = []
        self.an_indices = []
        self.en_indices = []
        self.in_indices = []
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

        # ─── Kuramoto coupling weights ───
        # All VNC neurons are GABA → negative coupling for anti-phase
        self.coupling_w = -self.weights.copy()  # W_ij < 0 → anti-phase

        # Scale coupling by indegree to normalize, then apply global scale
        self._normalize_coupling()
        self.coupling_w *= self.coupling_scale

        # ─── State ───
        self.phases = np.random.RandomState(42).uniform(0, 2 * math.pi, self.n).astype(np.float32)

        # Natural frequencies: spread around omega_base
        # IN neurons get base freq, DN slightly higher, EN slightly lower
        self.omegas = np.full(self.n, self.omega_base, dtype=np.float32)
        for idx in self.dn_indices:
            self.omegas[idx] = self.omega_base * 1.2  # DN faster
        for idx in self.en_indices:
            self.omegas[idx] = self.omega_base * 0.8  # EN slower
        # Add small random spread
        self.omegas += np.random.RandomState(42).uniform(-0.5, 0.5, self.n).astype(np.float32)

        # Rust SIMD backend (avx2/fma) — быстрый Kuramotostep
        self.use_rust = (tcm is not None and
                         hasattr(tcm, 'kuramoto') and
                         hasattr(tcm.kuramoto, 'kuramoto_step_sparse'))
        # Ensure arrays are C-contiguous for Rust backend
        self.phases = np.ascontiguousarray(self.phases)
        self.omegas = np.ascontiguousarray(self.omegas)
        self.coupling_w = np.ascontiguousarray(self.coupling_w)
        self.row_ptr = np.ascontiguousarray(self.row_ptr)
        self.col_indices = np.ascontiguousarray(self.col_indices)

        # Initialize tripod indices (None = no pairs found)
        self._tripod_a_idx = None
        self._tripod_b_idx = None
        self._top_recip_pairs = []

        # Find top reciprocal pairs for anti-phase detection
        self._find_recip_pairs()

        print(f"  [VNC.KUR] Loaded: {self.n:,} neurons, {self.n_syn:,} synapses [{time.time()-t0:.1f}s]")
        print(f"  [VNC.KUR] Groups: DN={len(self.dn_indices)} IN={len(self.in_indices)} "
              f"AN={len(self.an_indices)} EN={len(self.en_indices)}")
        print(f"  [VNC.KUR] Backend: {'🧊 Rust SIMD' if self.use_rust else '🐍 Python fallback'}")
        print(f"  [VNC.KUR] Omega: {self.omega_base:.1f} rad/s ({self.omega_base/(2*math.pi):.2f} Hz)")

    def _normalize_coupling(self):
        """Normalize coupling by number of incoming connections per neuron."""
        for i in range(self.n):
            s = int(self.row_ptr[i])
            e = int(self.row_ptr[i + 1])
            deg = e - s
            if deg > 0:
                # Scale so total coupling per neuron ≈ -1.0
                sum_w = abs(self.coupling_w[s:e]).sum()
                if sum_w > 0:
                    self.coupling_w[s:e] /= sum_w * 2.0  # total ≈ -0.5 per neuron

    def _find_recip_pairs(self, n_sample=200):
        """Find top reciprocal pairs within IN neurons."""
        in_set = set(self.in_indices[:n_sample])
        pairs = []
        for a in self.in_indices[:n_sample]:
            s = int(self.row_ptr[a])
            e = int(self.row_ptr[a + 1])
            for j in range(s, e):
                pre = int(self.col_indices[j])
                if pre in in_set and pre != a:
                    w_ab = float(self.coupling_w[j])
                    # Check reverse
                    s2 = int(self.row_ptr[pre])
                    e2 = int(self.row_ptr[pre + 1])
                    for k in range(s2, e2):
                        if int(self.col_indices[k]) == a:
                            w_ba = float(self.coupling_w[k])
                            gm = math.sqrt(abs(w_ab * w_ba))
                            pairs.append((a, pre, w_ab, w_ba, gm))
                            break
        pairs.sort(key=lambda x: -x[4])
        self._top_recip_pairs = pairs[:20]
        if len(pairs) >= 2:
            self._tripod_a_idx = pairs[0][0]
            self._tripod_b_idx = pairs[0][1]
            print(f"  [VNC.KUR] Top pair: IN[{pairs[0][0]}] ↔ IN[{pairs[0][1]}]  gm={pairs[0][4]:.0f}")

    def step(self, dn_drive=1.0, n_steps=10):
        """Run N Kuramoto steps.

        Args:
            dn_drive: descending drive amplitude (0.0-1.0), shifts DN omegas
            n_steps: number of Kuramoto integration steps
        """
        dt = self.dt_s

        # Apply DN drive as omega modulation
        if dn_drive > 0.01 and len(self.dn_indices) > 0:
            dn_shift = dn_drive * 2.0  # max 2 rad/s shift
            for idx in self.dn_indices:
                self.omegas[idx] = self.omega_base * 1.2 + dn_shift

        # Re-ensure contiguous (in case omegas were modified)
        if self.use_rust:
            try:
                for _ in range(n_steps):
                    tcm.kuramoto.kuramoto_step_sparse(
                        self.phases, self.omegas,
                        self.row_ptr, self.col_indices, self.coupling_w,
                        dt,
                    )
                # Rust modifies phases in-place — done!
                self.phases %= 2.0 * math.pi
                return
            except Exception as e:
                print(f"  ⚠ Kuramoto Rust error, falling back to Python: {e}")
                self.use_rust = False

        # ─── Python fallback ───
        sin_buf = np.zeros(self.n, dtype=np.float32)
        cos_buf = np.zeros(self.n, dtype=np.float32)
        coupling_buf = np.zeros(self.n, dtype=np.float32)

        for _ in range(n_steps):
            np.sin(self.phases, out=sin_buf)
            np.cos(self.phases, out=cos_buf)

            coupling_buf.fill(0.0)
            for i in range(self.n):
                s = int(self.row_ptr[i])
                e = int(self.row_ptr[i + 1])
                if s >= e:
                    continue
                pre_indices = self.col_indices[s:e]
                w = self.coupling_w[s:e]
                sin_diff = sin_buf[pre_indices] * cos_buf[i] - cos_buf[pre_indices] * sin_buf[i]
                coupling_buf[i] = np.dot(w, sin_diff)

            self.phases += dt * (self.omegas + coupling_buf)

        self.phases %= 2.0 * math.pi

    def get_phase(self, idx):
        """Get phase of neuron idx (0-2π)."""
        return self.phases[idx] % (2 * math.pi)

    def get_sin(self, idx):
        """Get sin(phase) of neuron idx, scaled to [-1, 1]."""
        return math.sin(self.phases[idx])

    def get_tripod_rates(self):
        """Get tripod gait rates: (tripod_A, tripod_B) in [0, 1].

        tripod_A = sin(phase of first IN in top pair) mapped to [0, 1]
        tripod_B = sin(phase of second IN in top pair) mapped to [0, 1]
        """
        if self._tripod_a_idx is None or self._tripod_b_idx is None:
            return 0.5, 0.5

        sa = math.sin(self.phases[self._tripod_a_idx])
        sb = math.sin(self.phases[self._tripod_b_idx])

        # Map from [-1, 1] to [0, 1] as leg activation
        rate_a = (sa + 1.0) * 0.5
        rate_b = (sb + 1.0) * 0.5

        return rate_a, rate_b

    def get_coherence(self):
        """Compute Kuramoto order parameter R (0=no sync, 1=full sync)."""
        sin_sum = np.sin(self.phases).sum()
        cos_sum = np.cos(self.phases).sum()
        r = math.sqrt(sin_sum**2 + cos_sum**2) / self.n
        return r

    def print_stats(self):
        """Print Kuramoto state diagnostics."""
        r = self.get_coherence()
        phase_a = self.get_phase(self._tripod_a_idx) if self._tripod_a_idx is not None else 0
        phase_b = self.get_phase(self._tripod_b_idx) if self._tripod_b_idx is not None else 0
        phase_diff = abs(phase_a - phase_b)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        print(f"  [VNC.KUR] Coherence R={r:.4f}  "
              f"Anti-phase diff={phase_diff:.2f} rad ({phase_diff/math.pi:.2f}π)")
        print(f"  [VNC.KUR] Tripod: A={phase_a:.2f} rad  B={phase_b:.2f} rad")
        if self.dn_indices:
            print(f"  [VNC.KUR] DN freq: {self.omegas[self.dn_indices].mean():.1f} rad/s")
