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
