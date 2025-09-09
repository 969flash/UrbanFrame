# -*- coding:utf-8 -*-
# r: networkx
try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import units, utils, road_network_generator, block_generator, get_road_data
import importlib

[
    importlib.reload(module)
    for module in [utils, units, road_network_generator, block_generator, get_road_data]
]


from road_network_generator import RoadNetworkGenerator
from block_generator import BlockGenerator


if __name__ == "__main__":
    # inputs setting
    generate = globals().get("GENERATE", [])
    print("Generate:", generate)
    if generate:
        road_data = get_road_data.get_road_data_from_layer()

        print("Road data: {} roads".format(len(road_data)))

        road_network = RoadNetworkGenerator().generate(road_data)
        blocks = BlockGenerator().generate(road_network)

        ########################################################
        ########################################################
        ########################################################
        # output
        edges = [e.curve for e in road_network.edges]
        nodes = [n.point for n in road_network.nodes]
        roads = [r.region for r in road_network.roads]
        junctions = [j.region for j in road_network.junctions]
        block_regions = [b.region for b in blocks]

        print(
            "RoadNetwork: {} edges, {} nodes, {} roads, {} junctions, {} blocks".format(
                len(edges), len(nodes), len(roads), len(junctions), len(blocks)
            )
        )
