#!/usr/bin/env python3
"""Bootstrap launcher for sardana_flow under the system Python.

The rebuild (`maturin develop`) installs the fresh `taas_checkmate.pyd` into the
project `.venv`; the system Python (which has torch+CUDA) has a stale copy. This
launcher prepends the venv site-packages so `taas_checkmate.pillars` resolves to
the fresh build while torch/numpy stay on the system install.

Usage:  python sardana_launch.py <main-args...>
Example:
    python sardana_launch.py train --steps 100 --n 256
    python sardana_launch.py generate --spec "motion detection for a drone"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".venv" / "Lib" / "site-packages"))

from sardana_flow.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
