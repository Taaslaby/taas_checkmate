#!/usr/bin/env python3
"""
FlyWire Codex API — test connection & explore basic connectome stats.
Использует CAVEclient для доступа к полному коннектому Drosophila.

Usage:
  python scripts/flywire_test.py
  python scripts/flywire_test.py --token YOUR_TOKEN
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

CAVE_SECRET_DIR = Path.home() / ".cloudvolume" / "secrets"
CAVE_SECRET_FILE = CAVE_SECRET_DIR / "global.daf-apis.com-cave-secret.json"

DATASET = "flywire_fafb_production"

def save_token(token: str):
    """Save token to cloud-volume secrets (standard CAVE auth location)."""
    CAVE_SECRET_DIR.mkdir(parents=True, exist_ok=True)
    secret = {"token": token}
    CAVE_SECRET_FILE.write_text(json.dumps(secret, indent=2))
    os.chmod(str(CAVE_SECRET_FILE), 0o600)
    print(f"  ✓ Token saved → {CAVE_SECRET_FILE}")

def test_connection(token: str | None = None):
    """Connect to FlyWire and explore the dataset."""
    if token:
        save_token(token)

    from caveclient import CAVEclient

    print(f"\n{'='*60}")
    print(f"  FlyWire Connectome Explorer")
    print(f"  Dataset: {DATASET}")
    print(f"{'='*60}\n")

    t0 = time.time()
    client = CAVEclient(DATASET)
    print(f"  ✓ Connected in {time.time()-t0:.1f}s")
    print(f"  ✓ CAVE version: {client.version}")

    # ── 1. Info about the dataset ──
    print(f"\n── Dataset info ──")
    try:
        info = client.info.info
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Image source: {info.get('image', {}).get('source', 'N/A')}")
        print(f"  Voxel resolution: {info.get('voxel_resolution', 'N/A')}")
    except Exception as e:
        print(f"  Info query failed: {e}")

    # ── 2. Available annotation tables ──
    print(f"\n── Available annotation tables (top 20) ──")
    try:
        tables = client.annotation.get_tables()
        for t in sorted(tables.keys())[:20]:
            desc = tables[t].get("description", "")
            print(f"  • {t:40s}  {desc[:60]}")
        if len(tables) > 20:
            print(f"  … and {len(tables)-20} more")
        print(f"  Total: {len(tables)} tables")
    except Exception as e:
        print(f"  Tables query failed: {e}")

    # ── 3. Query a quick neuron count ──
    print(f"\n── Neuron counts (nucleus_detection_v*) ──")
    try:
        nucleus_tables = [t for t in tables if "nucleus" in t.lower()]
        for ntable in sorted(nucleus_tables)[:5]:
            try:
                result = client.materialize.query_table(ntable, limit=1)
                total = client.materialize.live_query(
                    f"select count(*) from {ntable}"
                )
                if total is not None:
                    count = total.values[0][0] if hasattr(total, 'values') else len(total)
                else:
                    count = "?"
                print(f"  • {ntable:40s}  ~{count} rows")
            except Exception as e:
                print(f"  • {ntable:40s}  error: {str(e)[:50]}")
    except Exception as e:
        print(f"  Nucleus query failed: {e}")

    # ── 4. Cell types ──
    print(f"\n── Cell type tables ──")
    try:
        type_tables = [t for t in tables if "cell_type" in t.lower() or "type" in t.lower()]
        for tt in sorted(type_tables)[:5]:
            try:
                result = client.materialize.query_table(tt, limit=5)
                print(f"  • {tt:40s}  columns: {list(result.columns)[:8]}")
            except Exception as e:
                print(f"  • {tt:40s}  error: {str(e)[:50]}")
    except Exception as e:
        print(f"  Cell type query failed: {e}")

    # ── 5. Synapse table sample ──
    print(f"\n── Synapse tables ──")
    try:
        syn_tables = [t for t in tables if "synapse" in t.lower()]
        for st in sorted(syn_tables)[:3]:
            try:
                result = client.materialize.query_table(st, limit=3)
                print(f"  • {st:40s}  columns: {list(result.columns)[:10]}")
                print(f"    sample: {result.iloc[0].to_dict() if len(result) > 0 else 'empty'}")
            except Exception as e:
                print(f"  • {st:40s}  error: {str(e)[:50]}")
    except Exception as e:
        print(f"  Synapse query failed: {e}")

    # ── 6. Proofreading status ──
    print(f"\n── Proofreading status ──")
    try:
        status_tables = [t for t in tables if "status" in t.lower() or "proof" in t.lower()]
        for pt in sorted(status_tables)[:3]:
            try:
                result = client.materialize.query_table(pt, limit=5)
                print(f"  • {pt:40s}  columns: {list(result.columns)[:6]}")
            except Exception as e:
                print(f"  • {pt:40s}  error: {str(e)[:50]}")
    except Exception as e:
        print(f"  Status query failed: {e}")

    # ── 7. Materialization versions ──
    print(f"\n── Materialization versions ──")
    try:
        mv = client.materialize.get_versions()
        print(f"  Latest: {mv[-1] if mv else 'N/A'}")
        print(f"  Available: {mv[:5]}{'…' if len(mv) > 5 else ''}")
    except Exception as e:
        print(f"  Versions query failed: {e}")

    print(f"\n{'='*60}")
    print(f"  Done! Full explore above.")
    print(f"{'='*60}\n")

    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyWire Codex connection test")
    parser.add_argument("--token", type=str, default=None,
                        help="FlyWire API token (or set FLYWIRE_TOKEN env)")
    args = parser.parse_args()

    token = args.token or os.environ.get("FLYWIRE_TOKEN")
    if not token:
        # Если в secrets уже есть — используем его
        if CAVE_SECRET_FILE.exists():
            try:
                token = json.loads(CAVE_SECRET_FILE.read_text()).get("token")
                print("  Using existing token from secrets file")
            except Exception:
                pass

    if not token:
        print("ERROR: No token provided. Use --token TOKEN or set FLYWIRE_TOKEN env var.")
        sys.exit(1)

    test_connection(token)
