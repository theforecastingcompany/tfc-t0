"""The MLX runtime's tests run only where MLX installs (arm64 macOS, ``tfc-t0[mlx]``).

These tests also run on their own in the public repository, where the parent
``tests/conftest.py`` does not exist, so this file puts ``tools/`` (the benchmark
and checkpoint validator under test) on the import path itself.
"""

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

collect_ignore_glob = [] if importlib.util.find_spec("mlx") is not None else ["test_*.py"]
