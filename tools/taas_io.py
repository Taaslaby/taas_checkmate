"""Единый модуль чтения/записи бинарных форматов taas-checkmate.

До этого парсер .taas был скопирован в 8+ скриптах и разошёлся
(пропущенная проверка magic, отсутствующие receptor_types).
Единственная точка истины — здесь, рядом с Rust-реализацией
`crates/taas-format`.
"""

import struct

import numpy as np

TAAS_MAGIC = 0x54414153  # "TAAS"
HHP_MAGIC = 0x48485031   # "HHP1"
HAAP_MAGIC = 0x48414131  # "HAA1"
N_HAAP_PARAMS = 17


def load_taas(path):
    """Прочитать .taas (CSR-коннектом).

    Returns:
        (n_neurons, row_ptr, col_indices, values, nt_types, receptor_types)
        receptor_types — np.uint8[n_syn] или None, если блок отсутствует.
    """
    with open(path, "rb") as f:
        data = f.read()
    magic, _ver = struct.unpack_from("<II", data, 0)
    assert magic == TAAS_MAGIC, f"bad magic {magic:#x}"
    n_neurons, n_syn = struct.unpack_from("<QQ", data, 8)
    off = 24
    rp = np.frombuffer(data, dtype="<i4", count=n_neurons + 1, offset=off).astype(np.int32)
    off += 4 * (n_neurons + 1)
    ci = np.frombuffer(data, dtype="<i4", count=n_syn, offset=off).astype(np.int32)
    off += 4 * n_syn
    vals = np.frombuffer(data, dtype="<f4", count=n_syn, offset=off).astype(np.float32)
    off += 4 * n_syn
    nt = np.frombuffer(data, dtype="<f4", count=n_neurons, offset=off).astype(np.float32)
    off += 4 * n_neurons
    # optional receptor_types: 1 байт has_receptors + n_syn байт (коды из data.rs)
    receptor_types = None
    if off < len(data):
        has = data[off]
        off += 1
        if has == 1 and off + n_syn <= len(data):
            receptor_types = np.frombuffer(data, dtype="u1", count=n_syn, offset=off)
    return n_neurons, rp, ci, vals, nt, receptor_types


def write_taas(path, n_neurons, row_ptr, col_indices, values, nt_types,
               receptor_types=None, magic=TAAS_MAGIC, verbose=True):
    """Записать CSR в .taas формат чекмейта."""
    import os

    n_syn = len(col_indices)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", magic, 1))
        f.write(struct.pack("<QQ", n_neurons, n_syn))
        f.write(np.asarray(row_ptr, dtype="<i4").tobytes())
        f.write(np.asarray(col_indices, dtype="<i4").tobytes())
        f.write(np.asarray(values, dtype="<f4").tobytes())
        f.write(np.asarray(nt_types, dtype="<f4").tobytes())
        if receptor_types is not None and len(receptor_types) == n_syn:
            f.write(bytes([1]))
            f.write(np.asarray(receptor_types, dtype="u1").tobytes())
        else:
            f.write(bytes([0]))
    if verbose:
        print(f"  [save] {path}: {n_neurons} нейронов, {n_syn} синапсов, "
              f"{os.path.getsize(path)/1e6:.1f} MB")


def load_haap(path, n, n_params=N_HAAP_PARAMS):
    """Прочитать HAAP-физиологию: заголовок (магия "HAA1", n), далее n_params
    колонок по n float32. Returns: список колонок."""
    d = open(path, "rb").read()
    magic, nn = struct.unpack_from("<IQ", d, 0)
    assert magic == HAAP_MAGIC, f"bad haap magic {magic:#x}"
    assert nn == n, (hex(magic), nn, n)
    off = 12
    return [np.frombuffer(d, "<f4", count=n, offset=off + i * 4 * n).astype(np.float32)
            for i in range(n_params)]


def load_hhparams(path, n):
    """Прочитать HH-физиологию (магия HHP1, nn): 7 колонок по n float32.

    Returns: [g_na, g_k, g_l, e_na, e_k, e_l, inv_cm]
    """
    with open(path, "rb") as f:
        data = f.read()
    magic, nn = struct.unpack_from("<IQ", data, 0)
    assert magic == HHP_MAGIC and nn == n, (hex(magic), nn)
    off = 12
    cols = []
    for _ in range(7):
        cols.append(np.frombuffer(data, dtype="<f4", count=n, offset=off))
        off += 4 * n
    return cols


def load_mod(path, name):
    """Загрузить произвольный python-файл как модуль (для атласов/генераторов)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
