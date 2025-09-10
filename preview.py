import Rhino.Geometry as geo

try:
    from typing import List, Optional
except ImportError:
    pass
import utils, units
import importlib

importlib.reload(utils)
importlib.reload(units)
from units import Block, RoadNetwork, Node, Edge, Road, Junction


def block_id_preview_tags(blocks: "List[Block]", height: float = 10.0):
    """Return list of Rhino.Geometry.TextDot objects (preview only, not baked)."""
    if not blocks:
        return []
    dots = []
    for blk in blocks:
        reg = blk.region
        center = geo.AreaMassProperties.Compute(reg).Centroid
        if not utils.is_pt_inside(center, reg):
            center = utils.get_inside_check_pt(reg)

        label = str(blk.block_id)
        dot = geo.TextDot(label, center)
        dots.append(dot)
    return dots


road_network = globals().get("road_network", None)  # type: Optional[RoadNetwork]
blocks = globals().get("blocks", [])  # type: List[Block]
block_id_tags = block_id_preview_tags(blocks, height=10.0)
roads = [r.region for r in road_network.roads if r.region]
junctions = [j.region for j in road_network.junctions if j.region]
blocks = [b.region for b in blocks]
nodes = [n.point for n in road_network.nodes]
