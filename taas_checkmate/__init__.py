# taas_checkmate package initializer
import sys
from pathlib import Path

# Add root directory to sys.path so taas package is discoverable
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from .taas_checkmate import *
except ImportError:
    pass
