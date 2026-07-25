#!/usr/bin/env python3
"""
📦 Download Annotated Connectome Datasets

Скачивает аннотированные датасеты для fine-tuning Sardna control tokens.

Источники:
  1. FlyWire Annotations (GitHub) — cell types, NT, hemilineage, superclass
  2. FlyWire Connectivity (Zenodo) — synapse table (опционально, ~9.5 GB)
  3. FlyWire Skeletons (Zenodo) — морфология (~5.4 GB)
  4. Hemibrain (Xu et al. 2020) — центральный мозг дрозофилы
  5. C. elegans (OpenWorm) — полный коннектом
  6. MICrONS (mouse visual cortex) — функциональные типы

Usage:
  python scripts/download_datasets.py                     # всё лёгкое (< 500 MB)
  python scripts/download_datasets.py --all                # всё включая тяжелое
  python scripts/download_datasets.py --flywire-connectome # только FlyWire (~15 GB)
  python scripts/download_datasets.py --list               # только показать что есть
"""

import os
import sys
import argparse
import urllib.request
import json
import shutil
import gzip
from pathlib import Path

# ─── Пути ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "connectomes"
FLYWIRE_DIR = DATA_DIR / "flywire"
HEMIBRAIN_DIR = DATA_DIR / "hemibrain"
CELEGANS_DIR = DATA_DIR / "c_elegans"
MICRONS_DIR = DATA_DIR / "microns"

# ═══════════════════════════════════════════════════════════
#  DATASET REGISTRY
# ═══════════════════════════════════════════════════════════

DATASETS = {
    # ── 1. FlyWire Annotations (лёгкие, ~50 MB) ──
    "flywire_annotations": {
        "description": "FlyWire neuron annotations — cell types, NT, superclass, hemilineage",
        "size_mb": 50,
        "urls": [
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file2_connections.tsv",
        ],
        "files": [
            "flywire_neuron_annotations.tsv",
            "flywire_connections.tsv",
        ],
        "dir": FLYWIRE_DIR / "annotations",
        "skip_if_exists": True,
    },
    
    # ── 2. FlyWire Skeletons (тяжёлые, ~5.4 GB) ──
    "flywire_skeletons": {
        "description": "FlyWire neuron skeletons (3D morphology)",
        "size_mb": 5400,
        "urls": [
            "https://zenodo.org/records/10877326/files/sk_lod1_783_healed_ds2.parquet",
        ],
        "files": ["flywire_skeletons.parquet"],
        "dir": FLYWIRE_DIR / "morphology",
        "skip_if_exists": True,
        "heavy": True,
    },
    
    # ── 3. FlyWire Synapse Table (тяжёлый, ~9.5 GB) ──
    "flywire_synapses": {
        "description": "FlyWire full synapse table (proofread)",
        "size_mb": 9500,
        "urls": [
            "https://zenodo.org/records/10676866/files/flywire_synapses_783.feather",
            "https://zenodo.org/records/10676866/files/proofread_connections_783.feather",
            "https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy",
        ],
        "files": [
            "flywire_synapses_783.feather",
            "proofread_connections_783.feather",
            "proofread_root_ids_783.npy",
        ],
        "dir": FLYWIRE_DIR / "connectivity",
        "skip_if_exists": True,
        "heavy": True,
    },
    
    # ── 4. Hemibrain (Xu et al. 2020) — CSV дамп ──
    "hemibrain": {
        "description": "Hemibrain (Xu et al. 2020) — central brain connectome",
        "size_mb": 200,
        "urls": [
            "https://s3.amazonaws.com/hemibrain-release-v1.2/neuprint_export/neurons.csv.gz",
            "https://s3.amazonaws.com/hemibrain-release-v1.2/neuprint_export/synapses.csv.gz",
            "https://s3.amazonaws.com/hemibrain-release-v1.2/neuprint_export/connections.csv.gz",
        ],
        "files": [
            "hemibrain_neurons.csv.gz",
            "hemibrain_synapses.csv.gz",
            "hemibrain_connections.csv.gz",
        ],
        "dir": HEMIBRAIN_DIR,
        "skip_if_exists": True,
    },
    
    # ── 5. C. elegans (OpenWorm CElegansNeuroML + WormAtlas) ──
    "c_elegans": {
        "description": "C. elegans full connectome (OpenWorm CElegansNeuroML)",
        "size_mb": 5,
        "urls": [
            "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/herm_full_edgelist.csv",
            "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/CElegans/generated/wiring/morphology/neuron_soma_coords.csv",
        ],
        "files": [
            "c_elegans_edgelist.csv",
            "c_elegans_soma_coords.csv",
        ],
        "dir": CELEGANS_DIR,
        "skip_if_exists": True,
    },
    
    # ── 6. MICrONS — функциональные типы mice V1 ──
    "microns_functional": {
        "description": "MICrONS mouse visual cortex — functional cell types (легковесный запрос)",
        "size_mb": 10,
        "urls": [],  # через API
        "notes": "Требует CAVEclient. Запустите `pip install caveclient`",
        "dir": MICRONS_DIR,
        "skip_if_exists": True,
    },
}


# ═══════════════════════════════════════════════════════════
#  DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Скачать файл с прогресс-баром."""
    if dest.exists():
        print(f"  ✅ {dest.name} — already exists")
        return True
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  📥 Downloading {desc or dest.name}...")
    try:
        def report(block, blocksize, totalsize):
            if totalsize > 0:
                pct = min(100, block * blocksize / totalsize * 100)
                if block % 10 == 0:
                    print(f"    {pct:.0f}% ({totalsize/1e6:.0f} MB)", end="\r")
        
        urllib.request.urlretrieve(url, str(dest), reporthook=report)
        print(f"\n  ✅ {dest.name} — {dest.stat().st_size/1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def clone_flywire_annotations():
    """
    Альтернатива: клонировать весь репозиторий аннотаций.
    """
    dest = FLYWIRE_DIR / "annotations_git"
    if dest.exists():
        print(f"  ✅ FlyWire annotations repo already cloned")
        return
    
    print(f"\n  📥 Cloning flywire_annotations repository...")
    import subprocess
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/flyconnectome/flywire_annotations.git",
         str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✅ Cloned: {dest}")
    else:
        print(f"  ❌ Git clone failed: {result.stderr[:200]}")


def query_microns_cave():
    """Query MICrONS through CAVEclient."""
    print(f"\n  📥 Querying MICrONS via CAVEclient...")
    try:
        import caveclient
        client = caveclient.CAVEclient('minnie65_phase3')
        
        # Get cell types and functional properties
        query = """
        SELECT * FROM info__FunctionalProperties
        WHERE pre_pt_supervoxel_id IS NOT NULL
        LIMIT 1000
        """
        df = client.materialize.query(query)
        dest = MICRONS_DIR / "microns_functional_types.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"  ✅ MICrONS functional types: {len(df)} neurons -> {dest}")
        
        # Get synapse table sample
        query2 = """
        SELECT * FROM synapses_pni_2
        LIMIT 50000
        """
        df2 = client.materialize.query(query2)
        dest2 = MICRONS_DIR / "microns_synapses_sample.csv"
        df2.to_csv(dest2, index=False)
        print(f"  ✅ MICrONS synapses: {len(df2)} synapses -> {dest2}")
        
    except ImportError:
        print(f"  ⚠️  CAVEclient not installed. Run: pip install caveclient")
    except Exception as e:
        print(f"  ❌ MICrONS query failed: {e}")


# ═══════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════

def print_catalog():
    """Показать каталог доступных датасетов."""
    print(f"\n{'='*60}")
    print(f"  📦 CONNECTOME DATASET CATALOG")
    print(f"  Для fine-tuning Sardna control tokens")
    print(f"{'='*60}")
    
    for name, ds in DATASETS.items():
        heavy_tag = " ⚠️  HEAVY" if ds.get("heavy") else ""
        status = ""
        dest_dir = ds["dir"]
        if dest_dir.exists() and any(dest_dir.iterdir()):
            n_files = len(list(dest_dir.rglob("*")))
            total_size = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
            status = f" ✅ ({n_files} files, {total_size/1e6:.0f} MB)"
        
        print(f"\n  📍 {name}{heavy_tag}")
        print(f"     {ds['description']}")
        print(f"     Size: {ds['size_mb']} MB")
        print(f"     → {ds['dir']}{status}")
        
        if ds.get("notes"):
            print(f"     ℹ️  {ds['notes']}")
        
        for url, fname in zip(ds["urls"], ds["files"]):
            fpath = ds["dir"] / fname
            exists = "✅" if fpath.exists() else "⬜"
            print(f"     {exists} {fname}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="📦 Download annotated connectome datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --flywire-annotations
  python scripts/download_datasets.py --all
  python scripts/download_datasets.py --list
        """
    )
    parser.add_argument("--list", action="store_true",
                       help="Show catalog of available datasets")
    parser.add_argument("--all", action="store_true",
                       help="Download ALL datasets (including heavy ones)")
    parser.add_argument("--flywire-annotations", action="store_true",
                       help="Download FlyWire annotations only")
    parser.add_argument("--flywire-connectome", action="store_true",
                       help="Download FlyWire full connectome (~15 GB)")
    parser.add_argument("--hemibrain", action="store_true",
                       help="Download Hemibrain dataset")
    parser.add_argument("--celegans", action="store_true",
                       help="Download C. elegans connectome")
    parser.add_argument("--microns", action="store_true",
                       help="Query MICrONS functional types (needs CAVEclient)")
    args = parser.parse_args()
    
    # ── Show catalog ──
    if args.list:
        print_catalog()
        return
    
    # ── Default: lightweight datasets only ──
    if not any([args.all, args.flywire_annotations, args.flywire_connectome,
                args.hemibrain, args.celegans, args.microns]):
        args.flywire_annotations = True
        args.celegans = True
    
    if args.all:
        args.flywire_annotations = True
        args.flywire_connectome = True
        args.hemibrain = True
        args.celegans = True
        args.microns = True
    
    print(f"\n{'='*60}")
    print(f"  📦 DOWNLOADING CONNECTOME DATASETS")
    print(f"  Target: {DATA_DIR}")
    print(f"{'='*60}")
    
    # ── FlyWire Annotations ──
    if args.flywire_annotations:
        ds = DATASETS["flywire_annotations"]
        print(f"\n  🔗 FlyWire Annotations:")
        for url, fname in zip(ds["urls"], ds["files"]):
            download_file(url, ds["dir"] / fname, fname)
        # Также клонируем Git репозиторий
        clone_flywire_annotations()
    
    # ── FlyWire Connectome ──
    if args.flywire_connectome:
        ds = DATASETS["flywire_synapses"]
        print(f"\n  🔗 FlyWire Synapse Table (~9.5 GB):")
        for url, fname in zip(ds["urls"], ds["files"]):
            download_file(url, ds["dir"] / fname, fname)
        
        ds2 = DATASETS["flywire_skeletons"]
        print(f"\n  🔗 FlyWire Skeletons (~5.4 GB):")
        for url, fname in zip(ds2["urls"], ds2["files"]):
            download_file(url, ds2["dir"] / fname, fname)
    
    # ── Hemibrain ──
    if args.hemibrain:
        ds = DATASETS["hemibrain"]
        print(f"\n  🔗 Hemibrain (Xu et al. 2020):")
        for url, fname in zip(ds["urls"], ds["files"]):
            download_file(url, ds["dir"] / fname, fname)
    
    # ── C. elegans ──
    if args.celegans:
        ds = DATASETS["c_elegans"]
        print(f"\n  🔗 C. elegans (OpenWorm):")
        for url, fname in zip(ds["urls"], ds["files"]):
            download_file(url, ds["dir"] / fname, fname)
    
    # ── MICrONS ──
    if args.microns:
        print(f"\n  🔗 MICrONS (mouse visual cortex):")
        query_microns_cave()
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    total_files = 0
    total_size = 0
    for name, ds in DATASETS.items():
        dest = ds["dir"]
        if dest.exists():
            files = list(dest.rglob("*"))
            files = [f for f in files if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            total_files += len(files)
            total_size += size
            print(f"  {name:<25} {len(files):4d} files  {size/1e6:8.1f} MB")
    
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<25} {total_files:4d} files  {total_size/1e6:8.1f} MB")
    
    print(f"\n  📍 Data directory: {DATA_DIR}")
    print(f"  📥 Run with --all to download heavy datasets (>15 GB)")
    print(f"  ✅ Ready for Sardna fine-tuning!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
