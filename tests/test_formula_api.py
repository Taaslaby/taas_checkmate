#!/usr/bin/env python3
"""Smoke test: формула-модуль — сборка своего нейрона из ops в рантайме.

Запуск:
    cd taas_checkmate
    PYTHONPATH=target/debug python3 tests/test_formula_api.py

Ожидаемый результат: OK — все тесты проходят.
"""

import sys

# Windows-consoles default to cp1251 — force UTF-8 so the test runs on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np

passed = 0
failed = 0


def check(cond: bool, msg: str):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


def section(name: str):
    print(f"\n── {name} ─{'─' * max(0, 60 - len(name))}")


# ─── imports ─────────────────────────────────────────────

section("импорт и фасады")
try:
    from taas_checkmate.formula import (
        Neuron,
        Formula,
        FormulaPopulation,
        Trace,
        OpKind,
        OpNode,
        Reg,
        lif,
        run_program,
    )

    check(True, "from taas_checkmate.formula import …")
except Exception as e:  # noqa: BLE001
    check(False, f"from taas_checkmate.formula import … → {e}")
    sys.exit(1)

try:
    from taas_checkmate.pillars import Neuron as N2, Formula as F2, lif as lif2

    check(True, "фасад pillars: Neuron / Formula / lif")
    _n2 = lif2("FacadeLIF")
    check(isinstance(_n2, N2) and _n2.n_ops == 6, "pillars.lif() собирает Neuron")
except Exception as e:  # noqa: BLE001
    check(False, f"фасад pillars → {e}")

try:
    import taas_checkmate as tcm

    check(isinstance(tcm.Neuron, type) and callable(tcm.lif), "root: tcm.Neuron / tcm.lif")
    check(callable(tcm.run_program), "root: tcm.run_program")
except Exception as e:  # noqa: BLE001
    check(False, f"root-алиасы → {e}")


# ─── ручная сборка нейрона (dataflow) ────────────────────

section("Neuron — dataflow-сборка из ops")

n = Neuron("ManualLIF")
v = n.state("v", -70.0)
refr = n.state("refr", 0.0)
n.param("dt", 0.5)
inv_tau = n.param("inv_tau", 0.1)
v_rest = n.param("v_rest", -70.0)
v_thresh = n.param("v_thresh", -54.0)
v_reset = n.param("v_reset", -70.0)
tau_ref = n.param("tau_ref", 2.0)

zero = n.const_(0.0)
is_refr = n.cmp_gt(refr, zero)
raw = n.leak(v, n.I, inv_tau, v_rest)
gated = n.blend(is_refr, v, raw)
n.fire_reset(gated, v_thresh, v_reset, into=v)
n.refr_timer(refr, tau_ref, into=refr)

check(n.n_ops == 6, f"n_ops == 6 (получили {n.n_ops})")
check(n.n_states == 2 and n.n_params == 6, "n_states == 2, n_params == 6")
check("FireReset" in repr(n), "repr содержит FireReset")
check(n.ops[0].startswith("t0"), f"ops[0] — авто-темп: {n.ops[0]}")
check("i_syn" in repr(n.I), "n.I — регистр i_syn")

# регистры-обёртки
check(isinstance(v, Reg) and v.idx == 0, "Reg.idx == 0 для v")
check(isinstance(n.inv_tau, Reg), "n.inv_tau — Reg через __getattr__")

# OpKind enum
on = OpNode(OpKind.LEAK, 16)
check(on.kind == OpKind.LEAK and on.out_reg == 16, "OpKind enum + getter OpNode.kind/out_reg")


# ─── сырой слой OpNode / run_program ─────────────────────

section("сырой слой OpNode / run_program")

prog = [
    OpNode.const_(13, 0.1),     # t0 — inv_tau
    OpNode.const_(14, -54.0),   # t1 — v_thresh
    OpNode.const_(15, -70.0),   # t2 — v_reset
    OpNode(OpKind.LEAK, 16).with_in([0, 12, 13, 4]),   # s0 ← leak
    OpNode(OpKind.FIRERESET, 0).with_in([16, 14, 15]),  # s0 ← fire/reset
]
regs = [0.0] * 64
regs[0] = -70.0   # s0 = v
regs[4] = 0.5     # p0 = dt
regs[5] = -70.0   # p1 = v_rest
regs, sp = run_program(prog, regs, 0.5, 30.0)
check(isinstance(regs, list) and len(regs) == 64, "run_program вернул 64 регистра")
check(isinstance(sp, float) and sp in (0.0, 1.0), f"спайк-флаг {sp} ∈ {{0, 1}}")


# ─── simulate — быстрый прогон одного нейрона ────────────

section("Neuron.simulate — траектория и спайки")

tr = n.simulate(steps=200, i_syn=30.0)
check(isinstance(tr, Trace), "simulate возвращает Trace")
check(len(tr.v) == 200, f"len(tr.v) == 200 (получили {len(tr.v)})")
check(tr.spikes > 0, f"спайки при i_syn=30: {tr.spikes}")
check(tr.dt == 0.5 and tr.steps == 200, "dt/steps корректны")

tr0 = n.simulate(steps=200, i_syn=0.0)
check(tr0.spikes == 0, f"нет спайков при i_syn=0: {tr0.spikes}")

# v записывается после fire/reset — растёт от v_rest и приближается к порогу снизу
v_min, v_max = min(tr.v), max(tr.v)
check(v_min >= -70.5, f"v не уходит ниже v_reset: min={v_min:.1f}")
check(-60.0 < v_max <= -54.0, f"v приближается к порогу: max={v_max:.1f}")

# переопределение параметров: порог ниже → спайков больше
tr2 = n.simulate(steps=200, i_syn=30.0, params={"v_thresh": -60.0})
check(tr2.spikes > tr.spikes, f"порог ниже → спайков больше: {tr.spikes} → {tr2.spikes}")

# переопределение dt через params тоже работает (не только явным аргументом)
tr_dt = n.simulate(steps=100, i_syn=30.0, params={"dt": 1.0})
check(tr_dt.dt == 1.0 and tr_dt.spikes > 0, f"params dt=1.0 применился: {tr_dt.spikes} спайков")


# ─── ошибки валидации ────────────────────────────────────

section("валидация и ошибки")

def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False

bad = Neuron("bad")
bad.state("v", -70.0)
check(raises(lambda: bad.state("v", -70.0)), "дубликат состояния → ValueError")
bad2 = Neuron("bad2")
bad2.param("v_rest", -70.0)
check(raises(lambda: bad2.param("dt", 0.5)), "dt не первым → ValueError")

res = Neuron("res")
res.state("v", -70.0)
check(raises(lambda: res.state("I", 0.0)), "имя, занятое свойством билдера → ValueError")
check(raises(lambda: res.param("v", 0.0)), "param с именем состояния → ValueError")
check(raises(lambda: res.param("leak", 0.0)), "имя, занятое методом билдера → ValueError")
check(raises(lambda: bad.simulate(steps=10, i_syn=0.0)), "simulate без dt → ValueError")
check(raises(lambda: bad.simulate(steps=10, i_syn=0.0, params={"nope": 1.0})), "неизвестный параметр → ValueError")
check(raises(lambda: bad.compile()), "compile без dt → ValueError (validate)")


# ─── compile → Formula → FormulaPopulation (SIMD) ────────

section("compile → Formula → FormulaPopulation")

pre = lif("MyLIF")
model = pre.compile()
check(isinstance(model, Formula), "compile возвращает Formula")
check(model.name == "MyLIF", "model.name == 'MyLIF'")
check("v" in model.state_names and "inv_tau" in model.param_names, "имена состояний/параметров")

pop = model.population(1000, params={"v_thresh": -50.0})
check(isinstance(pop, FormulaPopulation), "population возвращает FormulaPopulation")
check(pop.n == 1000, "pop.n == 1000")

pop.set(v_rest=-70.0, v_reset=-60.0)
check(abs(pop.param("v_rest")[0] + 70.0) < 1e-6, "pop.set(v_rest=…) применился")

i_in = np.full(1000, 30.0, dtype=np.float32)
spikes = np.zeros(1000, dtype=np.int32)
total = 0
for _ in range(300):
    pop.step(i_in, spikes)
    total += int(spikes.sum())
check(total > 100, f"SIMD-популяция спайкает: {total} спайков за 300 шагов")

# per-neuron колонки
arr = np.full(1000, -50.0, dtype=np.float32)
pop.set_array("v_thresh", arr)
check(abs(pop.param("v_thresh")[0] + 50.0) < 1e-6, "set_array(v_thresh) применился")
st = np.full(1000, -65.0, dtype=np.float32)
pop.set_array("v", st)
check(abs(pop.state("v")[0] + 65.0) < 1e-6, "set_array на состояние v + pop.state() getter")

# repr
check("FormulaPopulation" in repr(pop) and "Formula<MyLIF>" in repr(model), "repr читаемые")

# ─── итог ────────────────────────────────────────────────

print()
if failed == 0:
    print(f"OK — все {passed} проверок прошли 🧠")
    sys.exit(0)
else:
    print(f"FAIL — {failed} из {passed + failed} проверок не прошли")
    sys.exit(1)
