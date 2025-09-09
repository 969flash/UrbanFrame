import Rhino.Geometry as geo

try:
    from typing import List, Optional
except ImportError:
    pass

from units import Block, RoadNetwork, Node, Edge, Road, Junction


def block_id_preview_tags(blocks: "List[Block]", height: float = 10.0):
    """Return list of Rhino.Geometry.TextDot objects (preview only, not baked)."""
    if not blocks:
        return []
    dots = []
    for blk in blocks:
        reg = getattr(blk, "region", None)
        if not reg:
            continue
        amp = geo.AreaMassProperties.Compute(reg)
        if not amp:
            continue
        center = amp.Centroid
        label = str(getattr(blk, "block_id", ""))
        if not label:
            continue
        try:
            dot = geo.TextDot(label, center)
            # TextDot doesn't take height; if relative size needed, user can scale later.
            dots.append(dot)
        except Exception:
            continue
    return dots


road_network = globals().get("road_network", None)  # type: Optional[RoadNetwork]
blocks = globals().get("blocks", [])  # type: List[Block]
block_id_tags = block_id_preview_tags(blocks, height=10.0)
roads = [r.region for r in road_network.roads if r.region]
junctions = [j.region for j in road_network.junctions if j.region]
blocks = [b.region for b in blocks]
