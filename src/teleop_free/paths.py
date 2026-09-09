"""Compatibility alias for the frozen scalebfm/ tree; the module lives in locomani (refactor step 4a)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # src/
from locomani import paths as _m

sys.modules[__name__] = _m
