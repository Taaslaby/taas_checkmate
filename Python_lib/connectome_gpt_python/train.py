#!/usr/bin/env python3
"""
connectome-gpt trainer v3 — быстрая загрузка, параллельная нарезка подграфов,
тренировка VAE на GPU, экспорт весов для Rust.
"""
import sys, os, time, math, csv, gzip, json, struct, random
from pathlib import Path
from collections import defaultdict, deque


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

FLYWIRE_DATA = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))
OUTPUT_DIR = Path("crates/connectome-gpt/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM = 10  # features above
HIDDEN_DIM = 256
LATENT_DIM = 32
SUBGRAPH_SIZE = 800
SUBGRAPHS_PER = 1000  # per connectome
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


# ── 1. Load connectome ──
def load_adj(conn_path):
    """Load connectome → adjacency lists (list of list of (neighbor, weight))."""
    t0 = time.time()
    pre_edges = defaultdict(list)
    pre_set = set()
    with gzip.open(str(conn_path), 'rt') as f:
        r = csv.reader(f)
        h = next(r)
        col = {c.strip().lower(): i for i, c in enumerate(h)}
        pre_c = col.get('pre_root_id', 0)
        post_c = col.get('post_root_id', 1)
        wt_c = col.get('syn_count', 3)
        for row in r:
            if len(row) <= max(pre_c, post_c):
                continue
            try:
                pre = int(row[pre_c]); post = int(row[post_c])
            except: continue
            pre_set.add(pre)
            wt = float(row[wt_c]) if len(row) > wt_c and row[wt_c].strip() else 1.0
            pre_edges[pre].append((post, wt))

    ids = sorted(pre_set)
    rid2i = {r: i for i, r in enumerate(ids)}
    n = len(ids)
    adj = [[] for _ in range(n)]
    for rid in ids:
        i = rid2i[rid]
        for post, wt in pre_edges[rid]:
            j = rid2i.get(post)
            if j is not None:
                adj[i].append((j, wt))
    print(f"  {conn_path.name}: {n:,} n, {sum(len(a) for a in adj):,} syn, "
          f"{time.time()-t0:.1f}s", flush=True)
    return adj, ids, n


def load_nt(conn_path, neurons_path, ids, n):
    """Load neurotransmitter types."""
    rid2i = {r: i for i, r in enumerate(ids)}
    nt = np.ones(n, dtype=np.float32)

    # From connections CSV nt_type column
    nt_votes = defaultdict(lambda: defaultdict(int))
    with gzip.open(str(conn_path), 'rt') as f:
        r = csv.reader(f)
        h = next(r)
        col = {c.strip().lower(): i for i, c in enumerate(h)}
        nt_c = col.get('nt_type', 4)
        pre_c = col.get('pre_root_id', 0)
        for row in r:
            if len(row) > nt_c and row[nt_c].strip():
                try: pre = int(row[pre_c])
                except: continue
                nt_votes[pre][row[nt_c].strip().upper()] += 1
    for rid, votes in nt_votes.items():
        idx = rid2i.get(rid)
        if idx is None: continue
        best = max(votes, key=votes.get)
        if 'GABA' in best or 'GLY' in best:
            nt[idx] = -1.0

    # From neurons CSV Predicted NT type
    if neurons_path and neurons_path.exists():
        with gzip.open(str(neurons_path), 'rt') as f:
            reader = csv.DictReader(f)
            fm = {c.strip().lower().replace(' ', '_'): c for c in reader.fieldnames}
            rc = fm.get('root_id', fm.get('root id', ''))
            nc = fm.get('predicted_nt_type', fm.get('nt_type', ''))
            if rc and nc:
                for row in reader:
                    try: rid = int(row[rc])
                    except: continue
                    val = row.get(nc, '').strip().upper()
                    if val and val not in ('', '?'):
                        idx = rid2i.get(rid)
                        if idx is not None:
                            nt[idx] = -1.0 if val == 'GABA' else 1.0
    return nt


# ── 2. Subgraph extraction (fast, no clustering/reciprocity) ──
def extract_subgraph_features(adj, nt, n, subgraph_size, seed):
    """BFS → compute 10 features. No O(n²) operations."""
    rng = random.Random(seed)

    # BFS (start from random node)
    start = rng.randint(0, n - 1)
    visited = {start}
    q = deque([start])
    while q and len(visited) < subgraph_size:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                if len(visited) < subgraph_size:
                    q.append(v)
                else:
                    break
    if len(visited) < subgraph_size:
        extras = rng.sample(range(n), min(subgraph_size - len(visited), n))
        visited.update(extras)

    nodes = list(visited)
    nn = len(nodes)
    if nn < 20:
        return None

    node_set = set(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    # Single pass: count edges, degree, weight sum, E/I
    out_deg = np.zeros(nn, dtype=np.int32)
    m = 0
    total_w = 0.0
    n_exc = 0
    n_inh = 0

    for u in nodes:
        cnt = 0
        for v, w in adj[u]:
            if v in node_set:
                cnt += 1
                total_w += abs(w)
        out_deg[node_to_idx[u]] = cnt
        m += cnt
        if nt[u] > 0:
            n_exc += 1
        else:
            n_inh += 1

    if m == 0:
        return None

    deg_f = out_deg.astype(np.float32)
    mean_deg = float(np.mean(deg_f))
    var_deg = float(np.var(deg_f)) if nn > 1 else 0.0

    return np.array([
        float(nn),
        float(m),
        m / max(nn * (nn - 1), 1) * 100.0,
        mean_deg,
        math.sqrt(var_deg / max(mean_deg, 1e-8)),
        min(n_exc / max(n_inh, 1), 10.0),
        total_w / m * 100.0,
        min(n_exc / max(n_inh, 1), 10.0),
        m / max(nn, 1) / 50.0,
        float(np.std(deg_f)) / max(mean_deg, 1e-8),
    ], dtype=np.float32)


# ── Dataset ──
class SubgraphDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.mean = self.features.mean(dim=0, keepdim=True)
        self.std = self.features.std(dim=0, keepdim=True).clamp(min=1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.normalize(self.features[idx])


# ── VAE ──
class GraphVAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv

    def loss_fn(self, recon, x, mu, lv, beta=1.0):
        mse = nn.functional.mse_loss(recon, x)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()
        return mse + beta * kl, mse, kl

    def export_weights(self, path):
        buf = bytearray()
        names = [('encoder', 0), ('encoder', 2), 'encoder_mu', 'encoder_logvar',
                 ('decoder', 0), ('decoder', 2), ('decoder', 4)]
        for name in names:
            if isinstance(name, tuple):
                layer = getattr(self, name[0])[name[1]]
            else:
                layer = getattr(self, name)
            for p in [layer.weight, layer.bias]:
                arr = p.data.cpu().numpy().flatten()
                buf.extend(struct.pack('<I', len(arr)))
                buf.extend(struct.pack(f'<{len(arr)}f', *arr))
        with open(path, 'wb') as f:
            f.write(buf)
        sz = len(buf)
        print(f"  💾 Weights: {path} ({sz} bytes)", flush=True)


# ── 4. Main ──
def main():
    print("═══ connectome-gpt v3 ═══\n", flush=True)
    t_start = time.time()

    conn_files = sorted(FLYWIRE_DATA.glob("connections_*.csv.gz"))
    print(f"Found {len(conn_files)} files, {SUBGRAPHS_PER} subgraphs each\n", flush=True)

    all_feats = []

    for f in conn_files:
        v = ''
        if '(' in f.name and ')' in f.name:
            v = f.name.split('(')[1].split(')')[0]
        np_path = FLYWIRE_DATA / f"neurons({v}).csv.gz" if v else FLYWIRE_DATA / "neurons.csv.gz"
        if np_path and not np_path.exists():
            np_path = None

        t0 = time.time()
        adj, ids, n = load_adj(f)
        nt = load_nt(f, np_path, ids, n)

        print(f"  Subgraphs: ", end='', flush=True)
        batch = []
        for s in range(SUBGRAPHS_PER):
            feat = extract_subgraph_features(adj, nt, n, SUBGRAPH_SIZE, s * 1000 + 42)
            if feat is not None:
                batch.append(feat)
            if s > 0 and s % 200 == 0:
                print(f"{s}", end=' ', flush=True)

        dt = time.time() - t0
        all_feats.extend(batch)
        print(f" → {len(batch)} valid, {dt:.0f}s", flush=True)

    if not all_feats:
        print("❌ No features extracted")
        return

    features = np.array(all_feats)
    valid = np.any(features > 0, axis=1)
    features = features[valid]
    print(f"\n📊 {len(features)} valid subgraphs, {np.sum(~valid)} dropped", flush=True)
    print(f"   Shape: {features.shape}", flush=True)

    # ── Train ──
    print(f"\n═══ Training VAE: {INPUT_DIM}→{HIDDEN_DIM}→{LATENT_DIM}→{HIDDEN_DIM}→{INPUT_DIM} ═══",
          flush=True)
    print(f"     Device: {device}  Batch: {min(256, len(features))}\n", flush=True)

    dataset = SubgraphDataset(features)
    loader = DataLoader(dataset, batch_size=min(256, len(features)), shuffle=True, num_workers=0)

    model = GraphVAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    best_loss = float('inf')
    patience = 100
    no_improve = 0

    for epoch in range(3000):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kl = 0.0
        nb = 0

        for batch_x in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, lv = model(batch_x)
            loss, mse, kl = model.loss_fn(recon, batch_x, mu, lv, beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()
            train_kl += kl.item()
            nb += 1

        avg_loss = train_loss / nb
        avg_mse = train_mse / nb
        avg_kl = train_kl / nb
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "vae_best.pt")
        else:
            no_improve += 1

        if epoch % 100 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            dt = time.time() - t_start
            print(f"  ep{epoch:4d}  loss={avg_loss:.4f}  mse={avg_mse:.4f}  "
                  f"kl={avg_kl:.4f}  best={best_loss:.4f}  lr={lr:.2e}  {dt:.0f}s",
                  flush=True)

        if no_improve >= patience:
            print(f"  ⏹  Early stop ep {epoch}", flush=True)
            break

    # Export
    model.load_state_dict(torch.load(OUTPUT_DIR / "vae_best.pt", map_location='cpu'))
    model.to('cpu')
    model.export_weights(OUTPUT_DIR / "vae_weights.bin")

    norm = {
        'mean': dataset.mean.cpu().numpy().tolist(),
        'std': dataset.std.cpu().numpy().tolist(),
        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM, 'latent_dim': LATENT_DIM,
    }
    with open(OUTPUT_DIR / "vae_norm.json", 'w') as f:
        json.dump(norm, f, indent=2)
    print(f"  💾 Norms: {OUTPUT_DIR / 'vae_norm.json'}", flush=True)

    model.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        fn = model.decode(z)
        f = dataset.unnormalize(fn)[0].cpu().numpy()
    print(f"\n═══ Test: n={f[0]:.0f} syn={f[1]:.0f} dens={f[2]:.3f}% "
          f"deg={f[3]:.1f} ei={f[5]:.1f} ═══", flush=True)
    print(f"Done in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
