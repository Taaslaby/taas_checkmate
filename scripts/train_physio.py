#!/usr/bin/env python3
"""
🎯 train_physio.py — градиентный спуск по физиологии MANC через движок.

Идея: параметры нейронов (g_na, g_k, g_l, e_l) — обучаемые. Каждая эпоха:
    1. заливаем параметры в движок через pyo3 (BrainRunner.set_hh_param_column)
    2. прогоняем симуляцию (forward через Rust)
    3. снимаем лосс по firing rates классов
    4. градиент по конечным разностям (движок = чёрный ящик)
    5. шаг градиентного спуска

Параметры: масштабы на КЛАСС нейронов (не per-neuron!):
    θ = [g_na_scale[C], g_k_scale[C], g_l_scale[C], e_l_offset[C]]  → 4·C параметров
Классы = коды nt_types в .taas (unk=0, optic=1, sensory=2, central=3,
motor=4, descending=5, ascending=6, endocr=7, visProj=8, visCent=9, sensAsc=10);
какие классы реально есть — определяется автоматически из .taas.

Лосс: log-MSE(fact_rate - target_rate) + λ·регуляризация к литературе
(λ·log²θ для масштабов, λ·θ² для сдвигов).

Градиент: --method spsa (2 прогона/эпоху, по умолчанию) или fd (58 прогонов).

Usage:
    python train_physio.py --taas resources/data/manc_vnc.taas \
        --epochs 60 --lr 0.03 --method spsa --steps 300 \
        --out resources/data/manc_physio_trained.hhparams
"""

import argparse
import os
import struct

import numpy as np

# ─── импорт движка ───
from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_taas

# HHParamIdx (сверено с pillars/src/zoo/neurons/hh.rs) — используем те, что меняем
COL_G_NA, COL_G_K, COL_G_LEAK, COL_E_LEAK = 1, 3, 5, 6

# полная карта классов 0..10 (коды nt_types из bake_brain.py SUPERCLASS_CODE)
CLASS_NAMES = {0: "unk", 1: "optic", 2: "sensory", 3: "central", 4: "motor",
               5: "desc", 6: "asc", 7: "endocr", 8: "visProj", 9: "visCent",
               10: "sensAsc"}
# какие классы реально есть в .taas — определяется в Engine.__init__
CLASSES = []
CLASS_POS = {}

# базовая литературная физиология по классам nt_types:
# g_na = 120 × HH_SUPERCLASS_SCALES, g_k = 36 × масштаб (generate_model_params.py),
# g_l и e_l — литературные (как в bake_physio.py)
BASE_PHYSIO = {
    0:  dict(g_na=120.0, g_k=36.0, g_l=0.10, e_l=-65.0),
    1:  dict(g_na=132.0, g_k=39.6, g_l=0.10, e_l=-65.0),  # optic ×1.1
    2:  dict(g_na=102.0, g_k=32.4, g_l=0.10, e_l=-60.0),  # sensory ×0.85
    3:  dict(g_na=120.0, g_k=36.0, g_l=0.10, e_l=-62.0),  # central ×1.0
    4:  dict(g_na=156.0, g_k=43.2, g_l=0.10, e_l=-65.0),  # motor ×1.3
    5:  dict(g_na=156.0, g_k=43.2, g_l=0.10, e_l=-65.0),  # descending ×1.3
    6:  dict(g_na=102.0, g_k=32.4, g_l=0.10, e_l=-60.0),  # ascending ×0.85
    7:  dict(g_na=84.0,  g_k=25.2, g_l=0.10, e_l=-65.0),  # endocrine ×0.7
    8:  dict(g_na=120.0, g_k=36.0, g_l=0.10, e_l=-62.0),  # visProj ×1.0
    9:  dict(g_na=120.0, g_k=36.0, g_l=0.10, e_l=-62.0),  # visCent ×1.0
    10: dict(g_na=102.0, g_k=32.4, g_l=0.10, e_l=-55.0),  # sensAsc ×0.85
}

# целевые firing rates (Гц) по классам при нашей стимуляции
TARGET_RATES = {0: 5.0, 1: 25.0, 2: 20.0, 3: 15.0, 4: 10.0, 5: 12.0,
                6: 15.0, 7: 5.0, 8: 15.0, 9: 15.0, 10: 15.0}


# .taas загрузка — единый парсер в tools/taas_io.py
# θ раскладка: 4 группы × число классов (определяется из .taas)


def theta_apply(theta, k, c):
    """θ[группа k, класс c] → плоский индекс."""
    return theta[k * len(CLASSES) + CLASS_POS[c]]


# ────────────────────────────────────────────────────────────────────
#  движок-обёртка
# ────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, taas_path, dt_ms=0.05, stim_frac=0.03, stim_amp=6.0, seed=7,
                 synaptic=False, poisson=False, poisson_rate=500.0, syn_norm=1.0,
                 stim_class=None):
        n, rp, ci, vals, nt, receptor_types = load_taas(taas_path)
        self.n = n
        self.nt = nt
        # какие классы реально есть в этом .taas (глобально — для theta/loss)
        global CLASSES, CLASS_POS
        CLASSES = [c for c in sorted(int(x) for x in np.unique(nt)) if c in CLASS_NAMES]
        CLASS_POS = {c: i for i, c in enumerate(CLASSES)}
        self.classes = CLASSES
        # values в FAFB — это СЧЁТЧИКИ синапсов между парами (медиана ~64),
        # а не проводимости: нормируем перед созданием ConnectomeData
        if syn_norm != 1.0:
            vals = (vals / syn_norm).astype(np.float32)
        self.conn = cb.ConnectomeData(rp, ci, vals, nt)
        # если в .taas запечены receptor_types — используем hh_synaptic
        # (тормозные GABA_A/GluCl у насекомых! без них сеть = чистый возбуждающий каскад)
        self.synaptic = synaptic and receptor_types is not None
        if self.synaptic:
            self.conn.set_receptor_types(receptor_types)
            self.runner = cb.BrainRunner(self.conn, model="hh_synaptic", dt_ms=dt_ms)
        else:
            self.runner = cb.BrainRunner(self.conn, model="hh", dt_ms=dt_ms)
        self.dt_ms = dt_ms

        # стимуляция: весь класс (stim_class) или случайный срез (не моторные)
        rng = np.random.RandomState(seed)
        if stim_class is not None:
            stim_ids = np.where(np.isin(nt, [float(c) for c in stim_class]))[0]
            self.stim_ids = stim_ids
        else:
            non_motor = np.where((nt != 4.0) & (nt > 0.0))[0]
            n_stim = max(1, int(stim_frac * len(non_motor)))
            self.stim_ids = rng.choice(non_motor, n_stim, replace=False)
        # Poisson-режим (спайк-поезда, физиологично для мозга) или константный ток
        self.poisson = poisson
        self.poisson_rate = poisson_rate
        self.ext_stim_amp = stim_amp
        self.ext = np.zeros(n, dtype=np.float32)
        if not poisson:
            self.ext[self.stim_ids] = stim_amp
        print(f"движок: {n} нейронов, стимулируем {len(self.stim_ids)}"
              f" [{'poisson ' + str(poisson_rate) + 'Hz' if poisson else 'const ' + str(stim_amp)}]")

        # индексы нейронов по классам
        self.class_idx = {c: np.where(nt == float(c))[0] for c in CLASSES}

    def apply_physio(self, theta):
        """θ (28, плоский) → per-neuron столбцы → движок."""
        g_na = np.empty(self.n, dtype=np.float32)
        g_k = np.empty(self.n, dtype=np.float32)
        g_l = np.empty(self.n, dtype=np.float32)
        e_l = np.empty(self.n, dtype=np.float32)
        for c in CLASSES:
            base = BASE_PHYSIO[c]
            mask = self.nt == float(c)
            g_na[mask] = base["g_na"] * theta_apply(theta, 0, c)
            g_k[mask] = base["g_k"] * theta_apply(theta, 1, c)
            g_l[mask] = base["g_l"] * theta_apply(theta, 2, c)
            e_l[mask] = base["e_l"] + theta_apply(theta, 3, c)
        # неизвестные (nt==0) — базовый
        self.runner.set_hh_param_column(COL_G_NA, g_na)
        self.runner.set_hh_param_column(COL_G_K, g_k)
        self.runner.set_hh_param_column(COL_G_LEAK, g_l)
        self.runner.set_hh_param_column(COL_E_LEAK, e_l)

    def forward(self, theta, n_steps):
        """Прогон движка с чистого состояния: reset → физиология → run.

        ⚠️ BrainRunner хранит состояние между вызовами run(), поэтому без
        reset() сеть «разогревается» за серию FD-прогонов и спайки дичают.
        reset() пересоздаёт HH-популяцию (стирая per-neuron параметры),
        поэтому физиология применяется ПОСЛЕ reset().
        """
        self.runner.reset()
        self.apply_physio(theta)
        if self.poisson:
            self.runner.clear_poisson_groups()
            self.runner.add_poisson_group(self.stim_ids.tolist(), self.poisson_rate, self.ext_stim_amp)
            buf = self.runner.run(n_steps, np.zeros(self.n, dtype=np.float32))
        else:
            buf = self.runner.run(n_steps, self.ext)
        spikes = np.asarray(buf).reshape(n_steps, self.n)
        rates = {}
        for c in CLASSES:
            idx = self.class_idx[c]
            if len(idx) == 0:
                rates[c] = 0.0
                continue
            cnt = spikes[:, idx].sum()
            rates[c] = cnt / len(idx) / (n_steps * self.dt_ms / 1000.0)
        return rates


# ────────────────────────────────────────────────────────────────────
#  лосс и градиент
# ────────────────────────────────────────────────────────────────────

def loss(engine, theta, n_steps, lam=1.0):
    rates = engine.forward(theta, n_steps)
    # лог-MSE: смягчает масштаб (208 Гц vs 12 Гц не доминирует над 1.4 vs 5)
    # и наказывает «смерть» нейрона (log(0+1)=0 при цели 12)
    L_data = np.mean([(np.log1p(rates[c]) - np.log1p(TARGET_RATES[c])) ** 2 for c in CLASSES])
    # регуляризация: log² для масштабов (θ=1 → 0), L2 для сдвигов (θ=0 → 0)
    reg = 0.0
    for k in range(3):  # масштабы g_na/g_k/g_l
        for c in CLASSES:
            reg += np.log(max(theta_apply(theta, k, c), 1e-3)) ** 2
    for c in CLASSES:  # сдвиг e_l
        reg += theta_apply(theta, 3, c) ** 2
    reg_term = lam * reg
    # ВАЖНО: рег-член входит в целевой функционал, иначе градиент по FD его не видит
    return L_data + reg_term, rates, reg_term


def finite_diff_grad(engine, theta, n_steps, eps=0.05, lam=1.0):
    """Градиент по конечным разностям: 2·28 прогонов движка."""
    grad = np.zeros_like(theta)
    L0, rates, _ = loss(engine, theta, n_steps, lam)
    for k in range(4):
        for c in CLASSES:
            idx = k * len(CLASSES) + CLASS_POS[c]
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[idx] += eps
            theta_m[idx] -= eps
            Lp, _, _ = loss(engine, theta_p, n_steps, lam)
            Lm, _, _ = loss(engine, theta_m, n_steps, lam)
            grad[idx] = (Lp - Lm) / (2 * eps)
    return grad, L0, rates


def spsa_grad(engine, theta, n_steps, c=0.1, lam=1.0, seed=None, n_samples=4):
    """Multi-sample SPSA-градиент: 2·n_samples прогонов движка на эпоху.

    Случайное возмущение Δ (Бернулли ±1) — несмещённая оценка градиента:
        ĝ = (L(θ + c·Δ) − L(θ − c·Δ)) / (2c) · (1/Δ)

    Усреднение по нескольким направлениям + momentum в цикле обучения
    сглаживают шум дискретного спайкового движка (одиночная SPSA взрывается).
    """
    rng = np.random.RandomState(seed)
    grad = np.zeros_like(theta)
    for _ in range(n_samples):
        delta = rng.choice([-1.0, 1.0], size=len(theta))
        theta_p = theta + c * delta
        theta_m = theta - c * delta
        Lp, _, _ = loss(engine, theta_p, n_steps, lam)
        Lm, _, _ = loss(engine, theta_m, n_steps, lam)
        grad += (Lp - Lm) / (2.0 * c) * (1.0 / delta)
    grad /= n_samples
    # baseline-прогон: L0 и rates при текущем θ (без лишнего прогона для печати)
    L0, rates, _ = loss(engine, theta, n_steps, lam)
    return grad, L0, rates


# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taas", default="resources/data/manc_vnc.taas")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=400)  # 400·0.05 = 20 мс
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--method", default="spsa", help="spsa (2 прогона/эпоху) | fd (58)")
    ap.add_argument("--c", type=float, default=0.1, help="SPSA perturbation size")
    ap.add_argument("--stim-frac", type=float, default=0.03, help="доля стимулируемых нейронов")
    ap.add_argument("--stim-amp", type=float, default=6.0, help="амплитуда стимула")
    ap.add_argument("--synaptic", action="store_true",
                    help="использовать hh_synaptic, если в .taas есть receptor_types (тормоза!)")
    ap.add_argument("--poisson", action="store_true",
                    help="стимулировать Poisson-спайк-поездами (вес = --stim-amp, rate = --poisson-rate)")
    ap.add_argument("--poisson-rate", type=float, default=500.0)
    ap.add_argument("--syn-norm", type=float, default=1.0,
                    help="нормировка values (счётчики синапсов → проводимости), FAFB: 256")
    ap.add_argument("--stim-class", default=None,
                    help="стимулировать весь класс целиком, напр. '2,8' (вместо случайного среза)")
    ap.add_argument("--out", default="resources/data/manc_physio_trained.hhparams")
    args = ap.parse_args()

    stim_class = None
    if args.stim_class:
        stim_class = [int(c) for c in args.stim_class.split(",")]
    engine = Engine(args.taas, stim_frac=args.stim_frac, stim_amp=args.stim_amp,
                    synaptic=args.synaptic, poisson=args.poisson,
                    poisson_rate=args.poisson_rate, syn_norm=args.syn_norm,
                    stim_class=stim_class)

    # θ0 = литература (все масштабы = 1, сдвиги = 0)
    n_cls = len(CLASSES)
    theta = np.concatenate([
        np.ones(n_cls),   # g_na_scale
        np.ones(n_cls),   # g_k_scale
        np.ones(n_cls),   # g_l_scale
        np.zeros(n_cls),  # e_l_offset
    ])
    print("классы:", {c: CLASS_NAMES[c] for c in CLASSES})
    print("цели (Гц):", TARGET_RATES)
    print()

    print(f"{'ep':>3} {'loss':>10} {'reg':>8}  " + "  ".join(f"{CLASS_NAMES[c]:>6}" for c in CLASSES))
    momentum = np.zeros_like(theta)
    for ep in range(args.epochs):
        if args.method == "spsa":
            # классическое SPSA-затухание: a_k = a/(k+1)^α, c_k = c/(k+1)^γ
            # шумная оценка градиента требует убывающих шагов, иначе θ улетает в clamp
            lr_k = args.lr / (ep + 1.0) ** 0.602
            c_k = args.c / (ep + 1.0) ** 0.101
            grad, L0, rates = spsa_grad(engine, theta, args.steps, c_k, args.lam,
                                        seed=1000 + ep, n_samples=4)
        else:
            lr_k = args.lr
            grad, L0, rates = finite_diff_grad(engine, theta, args.steps, args.eps, args.lam)
        # momentum сглаживает шум градиента (критично для SPSA на спайках)
        momentum = 0.9 * momentum + 0.1 * grad
        theta -= lr_k * momentum
        # ограничения: масштабы [0.05, 5], сдвиги [-80, 80]
        # (широкие — для плотных сетей вроде FAFB, где нужна сильная гиперполяризация)
        n_cls = len(CLASSES)
        theta[: 3 * n_cls] = np.clip(theta[: 3 * n_cls], 0.05, 5.0)
        theta[3 * n_cls :] = np.clip(theta[3 * n_cls :], -80.0, 80.0)
        # сброс momentum у границ: иначе параметр «давится» о стенку clampa
        at_scale_bound = (theta[: 3 * n_cls] == 0.05) | (theta[: 3 * n_cls] == 5.0)
        momentum[: 3 * n_cls][at_scale_bound] = 0.0
        at_off_bound = (theta[3 * n_cls :] == -80.0) | (theta[3 * n_cls :] == 80.0)
        momentum[3 * n_cls :][at_off_bound] = 0.0
        reg = 0.0
        for k in range(3):
            for c in CLASSES:
                reg += np.log(max(theta_apply(theta, k, c), 1e-3)) ** 2
        for c in CLASSES:
            reg += theta_apply(theta, 3, c) ** 2
        reg *= args.lam
        line = f"{ep:>3} {L0:>10.3f} {reg:>8.3f}  " + "  ".join(f"{rates[c]:>6.1f}" for c in CLASSES)
        print(line, flush=True)

    # финальная проверка
    Lf, rates_f, _ = loss(engine, theta, args.steps, args.lam)
    print(f"\nфинал: loss {L0:.3f} -> {Lf:.3f}")
    print("rates:", {CLASS_NAMES[c]: round(rates_f[c], 1) for c in CLASSES})

    # сохранить per-neuron физиологию (как .hhparams для bake_run)
    # e_na=+50 / e_k=-80 — литературные реверс-потенциалы (совпадают с BASE_PHYSIO E_K)
    g_na = np.empty(engine.n, dtype=np.float32)
    g_k = np.empty(engine.n, dtype=np.float32)
    g_l = np.empty(engine.n, dtype=np.float32)
    e_l = np.empty(engine.n, dtype=np.float32)
    for c in CLASSES:
        base = BASE_PHYSIO[c]
        mask = engine.nt == float(c)
        g_na[mask] = base["g_na"] * theta_apply(theta, 0, c)
        g_k[mask] = base["g_k"] * theta_apply(theta, 1, c)
        g_l[mask] = base["g_l"] * theta_apply(theta, 2, c)
        e_l[mask] = base["e_l"] + theta_apply(theta, 3, c)
    inv_cm = np.ones(engine.n, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<IQ", 0x48485031, engine.n))
        for col in (g_na, g_k, g_l, np.full(engine.n, 50.0, dtype=np.float32),
                    np.full(engine.n, -80.0, dtype=np.float32), e_l, inv_cm):
            f.write(col.astype("<f4").tobytes())
    print(f"\n[save] {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
