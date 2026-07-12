# 🧠 taas-checkmate — Python API

> Русскоязычная документация Python API.
> Движок: Rust + SIMD + CUDA, обёртка: PyO3 + numpy zero-copy.

## 📦 Сборка

```bash
cd taas_checkmate
cargo build --release --features python          # CPU only
cargo build --release --features python,cuda     # с GPU поддержкой
```

После сборки:
```bash
# Симлинк для import
ln -sf target/release/libtaas_checkmate.so target/release/taas_checkmate.so

# Использование
PYTHONPATH=target/release python3
```

## 🏗 Структура под-модулей

API организовано в под-модули. **Все функции также доступны на корневом уровне** для обратной совместимости:

| Модуль | Содержит |
|--------|----------|
| `taas_checkmate.neurons` | step_izhikevich, step_psi_wave, step_lif, stdp_update |
| `taas_checkmate.t3` | PyTriOscillator, PyNlmsDecoder, PyKalmanDecoder, t2spike_grad_* |
| `taas_checkmate.io` | safety_clamp, sigmoid_gate, t3_drive_encoder, echo_tap |
| `taas_checkmate.sparse` | compute_hamiltonian, step_psi_wave_full, propagate_spikes |
| `taas_checkmate.kuramoto` | kuramoto_step_sparse, phase_coherence, kalman_update |
| `taas_checkmate.snn` | PyLifNeuron, PySornNetwork, PyTtfsEncoder, bsa_encode |
| `taas_checkmate.eprop` | PyEPropLIF, PyEPropIzhikevichMulti, surrogate_gradient |
| `taas_checkmate.connectome` | bench_connectome_cpu, bench_connectome_gpu |
| `taas_checkmate.stp` | PyPoissonGenerator, PyShortTermPlasticity, compute_lfp |
| `taas_checkmate.rl` | PyActorCriticAgent |
| `taas_checkmate.koit` | xor_attention, rdrand_fill |
| `taas_checkmate.format` | save_taas, load_taas, inspect_taas |

Импорт — обоими способами:
```python
from taas_checkmate import step_izhikevich            # корневой уровень
from taas_checkmate.neurons import step_izhikevich    # под-модуль
from taas_checkmate.t3 import PyTriOscillator
```

---

## 🧬 1. Нейроны — `taas_checkmate.neurons`

### Izhikevich

```python
import numpy as np
from taas_checkmate.neurons import step_izhikevich

n = 1000
v = np.full(n, -65.0, dtype=np.float32)
u = np.full(n, -13.0, dtype=np.float32)
trace = np.zeros(n, dtype=np.float32)
a = np.full(n, 0.02, dtype=np.float32)
b = np.full(n, 0.2, dtype=np.float32)
c = np.full(n, -65.0, dtype=np.float32)
d = np.full(n, 2.0, dtype=np.float32)
I = np.random.randn(n).astype(np.float32) * 5 + 10
spikes = np.zeros(n, dtype=np.int32)

for _ in range(1000):
    step_izhikevich(v, u, trace, a, b, I, 0.5, 0.99, spikes, c, d)
    I[:] = 10.0  # reset current
    n_spikes = spikes.sum()
```

### ψ-wave (Schrödinger neuron)

```python
from taas_checkmate.neurons import step_psi_wave

n = 1000
v = np.cos(np.random.uniform(0, 2*np.pi, n)).astype(np.float32)  # Re(ψ)
u = np.sin(np.random.uniform(0, 2*np.pi, n)).astype(np.float32)  # Im(ψ)
h_real = np.zeros(n, dtype=np.float32)
h_imag = np.zeros(n, dtype=np.float32)
I_ext = np.zeros(n, dtype=np.float32)
speed = np.ones(n, dtype=np.float32)

for _ in range(100):
    # H·ψ сначала через compute_hamiltonian (из sparse)
    step_psi_wave(v, u, h_real, h_imag, I_ext, speed, 0.05, 1.0)
```

### STDP

```python
from taas_checkmate.neurons import stdp_update, stdp_homeostatic

weights = np.random.randn(1000).astype(np.float32) * 0.1
traces_pre = np.random.rand(1000).astype(np.float32)
traces_post = np.random.rand(1000).astype(np.float32)

stdp_update(weights, traces_pre, traces_post, 0.01, 0.005, -1.0, 1.0)
stdp_homeostatic(weights, traces_pre, traces_post, 0.01, 0.005, 50.0, 1.0)
```

---

## 🔄 2. T3 осцилляторы — `taas_checkmate.t3`

### Базовое использование

```python
from taas_checkmate.t3 import PyTriOscillator
import numpy as np

# Создать 64 осциллятора, частота 0.5–3.0 Гц
tri = PyTriOscillator(64, 0.5, 3.0)
tri.set_coupling(0.3, 0.15, 0.08)

# Один шаг → PST фичи
pst = tri.step(0.05)
print(f"PST dim: {len(pst)}")  # 18*64 + 12 = 1164

# N шагов → матрица PST
pst_matrix = tri.run(100, 0.05)
print(f"Shape: {pst_matrix.shape}")  # (100, 1164)

# Текущее состояние
r1, r2, r3 = tri.order_params()  # синхронизация
ph1 = tri.phases(1)               # фазы популяции 1
amp2 = tri.amplitudes(2)          # амплитуды популяции 2
pst_now = tri.pst()               # PST фичи сейчас
```

### Драйв (входной сигнал)

```python
# Драйв только для популяции 1
drive1 = np.sin(np.linspace(0, 2*np.pi, 64)).astype(np.float32)
tri.set_drive(drive1)
pst = tri.step(0.05)

# Драйв для всех 3 популяций
drive_all = np.random.randn(64 * 3).astype(np.float32)
tri.set_drive(drive_all)

# Драйв на шаге
pst = tri.step(0.05, drive=drive_all)
```

### Адаптивные декодеры

```python
from taas_checkmate.t3 import PyNlmsDecoder, PyKalmanDecoder

# NLMS — Normalized Least Mean Squares
dec = PyNlmsDecoder(1164, lr=0.1, eps=1e-8)
pred = dec.predict(pst)
error = dec.train(pst, 0.5)  # обучается на target
print(f"Prediction: {pred:.4f}, error: {error:.4f}")
print(f"Weights: {len(dec.weights())}")

# Kalman filter
kf = PyKalmanDecoder(1164, q=1.0, r=0.1)
pred = kf.predict(pst)
kf.train(pst, 0.5)
```

### T2-SPIKE-GRAD — дифференцируемый T3 (BPTT)

```python
from taas_checkmate.t3 import t2spike_grad_forward, t2spike_grad_backward, pst_linear_dim

n_osc, n_steps = 64, 100
pst_dim = pst_linear_dim(n_osc)

# Случайные драйвы
drives = np.random.randn(n_steps, 3 * n_osc).astype(np.float32)

# Forward: драйвы → PST + траектория
pst, traj = t2spike_grad_forward(
    n_osc, n_steps, 0.05, 10.0, 1.0,
    0.5, 3.0, 0.3, 0.15, 0.08,
    drives,
)
print(f"PST: {pst.shape}, traj: n_osc={traj.n_osc()}, n_steps={traj.n_steps()}")

# Backward: градиент PST → градиент драйвов
grad_pst = np.random.randn(pst_dim).astype(np.float32)
grad_drives = t2spike_grad_backward(grad_pst, traj)
print(f"Grad drives: {grad_drives.shape}")  # (100, 192)
```

### GPU версия (требует --features cuda)

```python
from taas_checkmate.t3 import t2spike_grad_forward_cuda, t2spike_grad_backward_cuda

pst, traj = t2spike_grad_forward_cuda(n_osc, n_steps, 0.05, 10.0, 1.0,
                                       0.5, 3.0, 0.3, 0.15, 0.08, drives)
grad_drives = t2spike_grad_backward_cuda(grad_pst, traj)
```

---

## 🔌 3. IO гейты — `taas_checkmate.io`

```python
from taas_checkmate.io import (safety_clamp, sigmoid_gate, phase_gate_apply,
                                 t3_drive_encoder, t3_complex_decoder,
                                 echo_tap, hebbian_update, thalamic_gate)

data = np.random.randn(100).astype(np.float32)
safety_clamp(data, -1.0, 1.0)  # обрезка

output = np.zeros(100, dtype=np.float32)
sigmoid_gate(output, data, gain=4.0, threshold=0.0)

# Фазовый гейт: модуляция амплитудой
r = np.abs(np.random.randn(100)).astype(np.float32)
phase_gate_apply(output, data, r, gain=2.0, threshold=0.5)

# Эхо-состояние
buffer = np.random.randn(64 * 256).astype(np.float32)
tap = np.zeros(64, dtype=np.float32)
echo_tap(tap, buffer, 64, 256, 0)

# Хеббианское обучение
weights = np.random.randn(100).astype(np.float32)
cos_diff = np.cos(np.random.randn(100)).astype(np.float32)
hebbian_update(weights, cos_diff, lr=0.01, decay=0.001, dt=0.5)
```

---

## 🔬 4. Разреженные операции — `taas_checkmate.sparse`

```python
from taas_checkmate.sparse import (compute_hamiltonian, step_psi_wave_full,
                                     propagate_spikes, apply_stdp_ltd)

# ψ-wave на коннектоме: H·ψ = W·(u + i·v)
v = np.cos(np.random.uniform(0, 2*np.pi, n)).astype(np.float32)
u = np.sin(np.random.uniform(0, 2*np.pi, n)).astype(np.float32)
h_real = np.zeros(n, dtype=np.float32)
h_imag = np.zeros(n, dtype=np.float32)
compute_hamiltonian(row_ptr, col_indices, values, v, u, h_real, h_imag)

# Полный ψ-wave шаг (гамильтониан + нормализация)
step_psi_wave_full(v, u, row_ptr, col_indices, values, 0.05, 1.0)

# Распространение спайков через CSR матрицу
propagate_spikes(row_ptr, col_indices, weights, spiked, nt_types, I_buf, 1.0)
```

---

## 📈 5. SNN классы — `taas_checkmate.snn`

**Свойства вместо copy_\*:** теперь можно читать состояние напрямую.

```python
from taas_checkmate.snn import PyLifNeuron

n = PyLifNeuron(1000, dt=0.5)
I = np.random.randn(1000).astype(np.float32) * 2 + 10
spikes = np.zeros(1000, dtype=np.int32)

for _ in range(100):
    n.step(I, spikes)

# Свойства — без pre-alloc!
membrane = n.v          # np.ndarray (1000,) — мембранный потенциал
threshold = n.theta     # np.ndarray (1000,) — адаптивный порог
```

```python
from taas_checkmate.snn import PySornNetwork, bsa_encode, phase_encode

# SORN — самоорганизующаяся рекуррентная сеть
sorn = PySornNetwork(256, w_init=0.1, w_target_sum=10.0)
sorn.step(spikes, traces_pre, traces_post)
sorn_weights = sorn.weights  # свойство!
sorn_theta = sorn.theta      # свойство!

# BSA кодирование
spikes_out = np.zeros(1000, dtype=np.int32)
signal = np.sin(np.linspace(0, 10*np.pi, 1000)).astype(np.float32)
kernel = bsa_kernel_hamming(32)
recon = np.zeros(1000, dtype=np.float32)
n_spikes = bsa_encode(spikes_out, signal, kernel, recon, threshold=0.5)
```

---

## 🎯 6. E-Prop — `taas_checkmate.eprop`

```python
from taas_checkmate.eprop import PyEPropLIF, PyEPropIzhikevichMulti, surrogate_gradient

# LIF с eligibility traces
eprop = PyEPropLIF(256, dt=1.0, tau_m=10.0, tau_e=20.0)
n_spikes = eprop.step(I, spikes_out, pre_spikes)

# Свойства
elig = eprop.eligibility  # np.ndarray
v = eprop.v                # np.ndarray

# Обновление весов через eligibility
eprop.update_weights(weights, learning_signals, lr=0.01)

# Surrogate gradient для обратного распространения
grad = surrogate_gradient(-55.0, -50.0, 2.0)
```

---

## 🧪 7. E-Prop Batch Trainer — `taas_checkmate.trainer`

```python
from taas_checkmate.trainer import PyEPropTrainer

# Многопоточный батч-трейнер для MNIST
trainer = PyEPropTrainer(
    ni=784, nh=256, no=10,
    n_steps=50, poisson_max_hz=200.0,
    dt=1.0, tau_e=30.0,
    lr_in=0.05, lr_out=0.02,
    bias=3.0, weight_scale=0.357,
    seed=42, num_workers=8,
)

# Обучение на батче
images = np.random.rand(64 * 784).astype(np.float32)
targets = np.random.randint(0, 10, 64).astype(np.int32)
avg_loss, correct, acc = trainer.train_batch(images, targets)

# Предсказание
pred = trainer.predict(images[:784])

# Веса
w_in = trainer.weights  # [ni × nh]
w_out = trainer.w_out   # [nh × no]
```

---

## 🔗 8. RL Agent — `taas_checkmate.rl`

```python
from taas_checkmate.rl import PyActorCriticAgent

agent = PyActorCriticAgent(n_input=4, n_hidden=256, n_actions=2)

# Шаг
action, value = agent.act(np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32))

# Обучение
td_error = agent.learn(reward=1.0, next_value=0.5, done=False)

# Параметры
print(f"Gamma: {agent.gamma}")
agent.gamma = 0.95

# Веса и eligibility через свойства
w_ih = agent.weights_ih     # input→hidden
w_ho = agent.weights_ho     # hidden→output
e_ho = agent.elig_ho        # hidden→output eligibility
```

---

## ⚡ 9. Poisson + STP — `taas_checkmate.stp`

```python
from taas_checkmate.stp import PyPoissonGenerator, PyShortTermPlasticity, compute_lfp

# Poisson генератор
gen = PyPoissonGenerator(1000, dt=1.0, seed=42)
gen.set_rate(0, 50.0)  # нейрон 0: 50 Гц
spikes = np.zeros(1000, dtype=np.int32)
gen.step(spikes)

# Short-Term Plasticity (Tsodyks-Markram)
stp = PyShortTermPlasticity(1000, dt=1.0)
I_out = np.zeros(1000, dtype=np.float32)
stp.step(spikes, I_out)

# Состояние STP
x = stp.x  # депрессия (свойство)
u = stp.u  # фасилитация (свойство)
```

---

## 🧠 10. Connectome Benchmark — `taas_checkmate.connectome`

```python
from taas_checkmate.connectome import bench_connectome_cpu, bench_connectome_gpu

# CPU (SIMD + rayon)
elapsed, total_spikes = bench_connectome_cpu(
    row_ptr, col_indices, values,
    n_steps=500, dt=0.5,
    v, u, trace, a, b, c, d,
    nt_types, external_I, I_buf, spikes,
)
print(f"CPU: {elapsed:.3f}s, {total_spikes:,} spikes")

# GPU (CUDA)
elapsed, total_spikes = bench_connectome_gpu(
    row_ptr, col_indices, values,
    n_steps=500, dt=0.5,
    v, u, a, b, c, d,
    nt_types, external_I, spikes,
)
print(f"GPU: {elapsed:.3f}s, {total_spikes:,} spikes")
```

---

## 🔢 11. XOR Attention + RNG — `taas_checkmate.koit`

```python
from taas_checkmate.koit import xor_attention, rdrand_fill

# Бинарная attention через XOR + popcount
query = np.random.randint(0, 256, 16).astype(np.uint8)
keys = np.random.randint(0, 256, 32 * 16).astype(np.uint8)
values = np.random.randn(32).astype(np.float32)
out = np.zeros(32, dtype=np.float32)
xor_attention(query, keys, values, out, n_keys=32, n_bytes=16)

# Аппаратный RNG
buf = np.zeros(1000, dtype=np.float32)
rdrand_fill(buf)
```

---

## 💾 12. Формат .taas — `taas_checkmate.format`

```python
from taas_checkmate.t3 import PyTriOscillator, PyNlmsDecoder
from taas_checkmate.format import save_taas, load_taas, inspect_taas

# Сохранить состояние
tri = PyTriOscillator(64, 0.5, 3.0)
dec = PyNlmsDecoder(1164, 0.1, 1e-8)
save_taas("model.taas", tri=tri, nlms=dec)

# Загрузить
state = load_taas("model.taas")
tri = state["tri"]
dec = state["nlms"]
print(state["_meta"])  # метаданные

# Инспектировать
print(inspect_taas("model.taas"))
```

---

## 🚀 Примеры исследований

См. `research/python_explorer/` для полных скриптов:

| Скрипт | Что делает |
|--------|------------|
| `psi_wave_fafb.py` | ψ-wave на 139K нейронов FAFB коннектома |
| `psi_eeg.py` | EEG-подобные сигналы из ψ-wave |
| `pci.py` | Perturbational Complexity Index (Casali 2013) |
| `narma10_t3.py` | T3 reservoir vs ESN на NARMA10 |
| `cartpole_t3.py` | CartPole × T3 + NLMS + REINFORCE |
| `cartpole_t3_bptt.py` | CartPole × T2-SPIGRAD (BPTT через PyTorch) |
| `bench_connectome.py` | FlyWire 138K: CPU (SIMD) vs GPU (CUDA) |
| `tune_eprop_mnist.py` | E-Prop Izhikevich на MNIST с grid search |

---

## 📐 Сборка с CUDA

```bash
cargo build --release --features python,cuda

# Python: загружаем напрямую (не из site-packages)
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('taas_checkmate',
    'target/release/libtaas_checkmate.so')
t3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t3)
print('CUDA backward:', hasattr(t3, 't2spike_grad_backward_cuda'))
"
```
