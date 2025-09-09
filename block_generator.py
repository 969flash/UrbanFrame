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
import scriptcontext as sc
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
        """Road/Junction 영역 합집합에서 내부(블록) 영역 추출."""
        if not road_network:
            return []
        road_regions = [r.region for r in road_network.roads if r.region]
        junction_regions = [j.region for j in road_network.junctions if j.region]

        all_regions = road_regions + junction_regions

        all_regions = utils.get_regions_from_crvs(all_regions)

        edge_crvs = [e.curve for e in road_network.edges]

        # all_regions 중 edge_crvs와 겹치는 영역 제거
        if edge_crvs:
            # edge_crvs와 교차하지 않는 영역만 유지
            filtered = []
            for r in all_regions:
                if not any(utils.has_intersection(r, ec) for ec in edge_crvs):
                    filtered.append(r)
            all_regions = filtered

        blocks = []
        for i, r in enumerate(all_regions):
            block = Block(r, block_id=i + 1)
            # 블록에 인접한 도로/교차로 찾기
            for road in road_network.roads:
                if utils.has_intersection(r, road.region):
                    block.roads.append(road)
            for junction in road_network.junctions:
                if utils.has_intersection(r, junction.region):
                    block.junctions.append(junction)
            blocks.append(block)

        return blocks
