"""Direct-file import of _class_catalogues.py, bypassing perception.models's
package __init__ chain.

_class_catalogues.py's own docstring says it's deliberately import-light (no
torch/transformers) so lightweight callers can pull the name tuples cheaply
-- but a plain `from perception.models.semantic._class_catalogues import
...` still executes `perception/models/__init__.py` first (Python always
runs parent-package __init__ on any submodule import), which eagerly pulls
in the full instance/factory/YOLO stack (ultralytics, requests, etc.) --
exactly the heavy chain this module's own docstring says it's avoiding, and
enough to break a minimal-dependency venv (e.g. .venv-sam3, which has no
`requests`) for what should be a handful of plain string tuples.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_CATALOGUES_PATH = Path(__file__).resolve().parents[2] / "src/perception/models/semantic/_class_catalogues.py"
_spec = importlib.util.spec_from_file_location("_class_catalogues", _CATALOGUES_PATH)
_class_catalogues = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_class_catalogues)

SAM3_FINEGRAINED_NAMES: tuple[str, ...] = _class_catalogues.SAM3_FINEGRAINED_NAMES
COARSE_CLASSES: tuple[str, ...] = _class_catalogues.COARSE_CLASSES
FINE_TO_COARSE: dict[str, str] = _class_catalogues.FINE_TO_COARSE
