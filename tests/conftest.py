"""Make the model subdirectories importable from the tests.

The project ships as a set of scripts rather than an installed package, so
the per-model directories are added to sys.path here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in (ROOT, ROOT / "wilson_cowan", ROOT / "liquidity"):
    sys.path.insert(0, str(sub))
