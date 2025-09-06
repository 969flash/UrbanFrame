# -*- coding:utf-8 -*-
try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import units, utils, road_network_generator, block_generator
import importlib

[
    importlib.reload(module)
    for module in [utils, units, road_network_generator, block_generator]
]


from road_network_generator import RoadNetworkGenerator
from block_generator import BlockGenerator


if __name__ == "__main__":
    # inputs setting
    road_data = globals().get("road_data", [])

    rdnetwork_generator = RoadNetworkGenerator()
    road_network = rdnetwork_generator.generate(road_data)

    edges = [e.curve for e in road_network.edges]
    nodes = [n.point for n in road_network.nodes]
    roads = [r.region for r in road_network.roads]
    junctions = [j.region for j in road_network.junctions]
    print(
        "RoadNetwork: {} edges, {} nodes, {} roads, {} junctions".format(
            len(edges), len(nodes), len(roads), len(junctions)
        )
    )
