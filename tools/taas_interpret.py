#!/usr/bin/env python3
"""
taas_interpret.py — pure-Python (no numpy) interpretability analysis of
generated connectome .taas files (sardaanaCM / generative connectome).

Reads the TAAS1 format: [magic 5B]["TAAS1"] [u64 LE header_len] [JSON header] [tensor data]
Computes graph-theoretic + functional-interpretability statistics.
"""
import struct, json, zlib, sys, math
from collections import defaultdict

TAAS_MAGIC = b"TAAS1"

def load_taas(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:5] == TAAS_MAGIC, f"bad magic: {data[:5]!r}"
    hlen = struct.unpack("<Q", data[5:13])[0]
    header = json.loads(data[13:13+hlen].decode("utf-8"))
    tdata = data[13+hlen:]
    comps = {}
    for c in header.get("components", []):
        comps[c["name"]] = c
    return header, comps, tdata

def get_tensor(comps, tdata, name):
    c = comps.get(name)
    if c is None:
        return None
    raw = tdata[c["offset"]:c["offset"]+c["size"]]
    if c.get("compression") == "zstd":
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(raw)
    dt = {"f32": "f", "i32": "i", "i64": "q", "u64": "Q", "u8": "B"}.get(c["dtype"])
    n = math.prod(c["shape"])
    return list(struct.unpack("<" + dt * n, raw))

def degree_stats(deg):
    if not deg:
        return dict(mean=0.0, median=0.0, std=0.0, max=0, min=0, cv=0.0, hubs=0)
    m = sorted(deg)
    n = len(m)
    mean = sum(m)/n
    med = m[n//2] if n % 2 else 0.5*(m[n//2-1]+m[n//2])
    var = sum((x-mean)**2 for x in m)/n
    std = var**0.5
    # hubs: degree > mean + 2*std
    hubs = sum(1 for x in m if x > mean + 2*std)
    return dict(mean=round(mean,2), median=med, std=round(std,2), max=max(m),
                min=min(m), cv=round(std/max(mean,1e-9),3), hubs=hubs)

def clustering_coeff(adj):
    """avg local clustering (directed-unaware)"""
    total, valid = 0.0, 0
    for i, nbrs in enumerate(adj):
        k = len(nbrs)
        if k < 2:
            continue
        nset = set(nbrs)
        edges = sum(1 for a in nbrs for b in adj[a] if b in nset)
        total += edges / (k*(k-1))
        valid += 1
    return total/max(valid,1)

def reciprocity(adj):
    """True reciprocity: share of edges i->j that have a reverse edge j->i.

    Uses reverse-index sets built from the SAME adjacency (correct); the old
    `i in rev[j]` check was trivially true because rev[j] is built from the
    same edges being tested.
    """
    rev = [set() for _ in range(len(adj))]
    for i in range(len(adj)):
        for j in adj[i]:
            rev[j].add(i)
    total = recip = 0
    for i in range(len(adj)):
        for j in adj[i]:
            total += 1
            if j in rev[i]:  # reverse edge j->i exists
                recip += 1
    return recip/max(total,1)

def avg_path_len(adj, n, limit=500):
    if n > limit:
        return None
    INF = float("inf")
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
        for j in adj[i]:
            dist[i][j] = 1.0
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF: continue
            di = dist[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    total = cnt = 0
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < INF:
                total += dist[i][j]; cnt += 1
    return round(total/max(cnt,1), 3)

def analyze(path):
    header, comps, tdata = load_taas(path)
    meta = header.get("metadata", {})
    row_ptr = get_tensor(comps, tdata, "row_ptr")
    col = get_tensor(comps, tdata, "col_indices")
    vals = get_tensor(comps, tdata, "values")
    nt = get_tensor(comps, tdata, "nt_types")
    depth = get_tensor(comps, tdata, "depth")

    n = int(meta.get("n_neurons", len(row_ptr)-1))
    m = len(col)
    adj = [[] for _ in range(n)]
    rev = [[] for _ in range(n)]
    for i in range(n):
        s, e = row_ptr[i], row_ptr[i+1]
        for k in range(s, e):
            j = col[k]
            if 0 <= j < n:
                adj[i].append(j)
                rev[j].append(i)

    outdeg = [len(a) for a in adj]
    ds = degree_stats(outdeg)
    density = m / (n*(n-1)) if n > 1 else 0.0
    n_exc = sum(1 for x in nt if x > 0) if nt else 0
    n_inh = sum(1 for x in nt if x < 0) if nt else 0
    ei = round(n_exc/max(n_inh,1), 3)
    wabs = [abs(v) for v in vals]
    wmean = round(sum(wabs)/max(m,1), 4)
    wvar = sum((x-wmean)**2 for x in wabs)/max(m,1)
    recip = round(reciprocity(adj), 4)
    clust = round(clustering_coeff(adj), 4)
    apl = avg_path_len(adj, n)
    # small-world proxy
    sw = None
    if apl and ds["mean"] > 1:
        rand_clust = density
        rand_path = math.log(n)/math.log(ds["mean"])
        sw = round((clust/max(rand_clust,1e-10))/(apl/max(rand_path,0.01)), 3)

    # degree distribution histogram (log-bins for interpretability)
    hist = defaultdict(int)
    for d in outdeg:
        if d == 0:
            hist["0"] += 1
        else:
            b = 2**int(math.log2(d))
            hist[str(b)] += 1
    dist_hist = dict(sorted(hist.items(), key=lambda kv: int(kv[0]) if kv[0]!="0" else 0))

    # weight sign balance
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    wp = round(n_pos/max(n_neg,1), 3)

    return dict(
        source=meta.get("source", meta.get("type","?")),
        n=n, m=m,
        density=round(density, 6),
        degree=ds,
        ei_ratio=ei,
        weight_mean=wmean, weight_std=round(math.sqrt(wvar),4),
        weight_pos_neg=wp,
        reciprocity=recip,
        clustering=clust,
        avg_path_length=apl,
        small_world=sw,
        degree_hist=dist_hist,
        n_exc=n_exc, n_inh=n_inh,
        depth_std=round(float(sum((x- sum(depth)/len(depth))**2 for x in depth)/len(depth))**0.5, 4) if depth and len(depth) else None,
    )

def main():
    files = sys.argv[1:] or [
        r"C:\Users\anton\Desktop\Проекты\asm_engines\sardaanaCM\experiments\sardna_ultra\generated\flywire_10k_n50_t0_85.taas",
        r"C:\Users\anton\Desktop\Проекты\asm_engines\sardaanaCM\experiments\sardna_ultra\generated\flywire_10k_n200_t0_85.taas",
        r"C:\Users\anton\Desktop\Проекты\asm_engines\sardaanaCM\experiments\sardna_ultra\generated\flywire_10k_n500_t0_85.taas",
    ]
    for p in files:
        print(f"\n{'='*70}\nFILE: {p.split(chr(92))[-1]}")
        r = analyze(p)
        print(f"  source: {r['source']}  |  N={r['n']}  synapses={r['m']}  density={r['density']}")
        print(f"  degree: mean={r['degree']['mean']} median={r['degree']['median']} "
              f"std={r['degree']['std']} max={r['degree']['max']} CV={r['degree']['cv']} hubs={r['degree']['hubs']}")
        print(f"  E/I ratio: {r['ei_ratio']}  ({r['n_exc']} exc / {r['n_inh']} inh)")
        print(f"  weights: mean|w|={r['weight_mean']} std={r['weight_std']} pos/neg={r['weight_pos_neg']}")
        print(f"  reciprocity={r['reciprocity']}  clustering={r['clustering']}  "
              f"APL={r['avg_path_length']}  SW={r['small_world']}")
        print(f"  degree histogram (2^k bins): {r['degree_hist']}")
        print(f"  depth_std={r['depth_std']}")

if __name__ == "__main__":
    main()
