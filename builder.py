# -*- coding:utf-8 -*-
"""
Grasshopper runner for a single building, mirroring previous main.py usage.
Place this script in a GhPython component and wire inputs accordingly.
"""

try:
    from typing import List
except ImportError:
    pass
import block_generator as bg
import importlib

importlib.reload(bg)

if __name__ == "__main__":
    # Build inputs directly from GH globals and run
    inputs = bg.FGInputs.from_globals(globals())
    generator = bg.FacadeGenerator(inputs)
    _pattern_type = int(globals().get("pattern_type", 1))
    facades = generator.generate(_pattern_type)

    # Outputs for GH
    glasses, walls, frames, slabs = _flatten(facades)
