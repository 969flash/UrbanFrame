# -*- coding:utf-8 -*-
"""
Grasshopper runner for a single building, mirroring previous main.py usage.
Place this script in a GhPython component and wire inputs accordingly.
"""

try:
    from typing import List
except ImportError:
    pass
import Rhino.Geometry as geo
import units
import utils
import importlib

importlib.reload(utils)
importlib.reload(units)

from units import Node, Edge, Road, Junction, RoadNetwork, Block


class BlockGenerator:
    def __init__(self):
        pass

    def generate(self, road_network: RoadNetwork) -> List[Block]:
        """블록을 생성합니다."""
        blocks = []
        
        return blocks