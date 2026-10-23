"""The ONNX export tests run only where the ``onnx`` dependency group installs (Python 3.11+).

This file puts ``tools/`` (the exporter and validator under test) on the import path.
"""

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_REQUIRED = ("onnx", "onnxruntime", "onnxscript", "torch")
collect_ignore_glob = [] if all(importlib.util.find_spec(name) for name in _REQUIRED) else ["test_*.py"]
