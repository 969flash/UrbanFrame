try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp
import utils
import importlib
import scriptcontext as sc

importlib.reload(utils)
TOL = 5


class Edge:
    def __init__(self, curve: geo.Curve, width: float):
        self.curve = curve  # type: geo.Curve
        self.width = width  # type: float

    def is_at(self, node: "Node") -> bool:
        return (
            self.curve.PointAtStart == node.point or self.curve.PointAtEnd == node.point
        )


class Node:
    def __init__(self, point: geo.Point3d):
        self.point = point
        self.edges = []  # type: List[Edge]

    def add_edges(self, edges: List[Edge]) -> None:
        self.edges.extend(edges)

    def _key_from_point(self, p: geo.Point3d):
        """Quantize point to TOL-sized grid so equality and hashing align."""
        q = TOL
        return (
            int(round(p.X / q)),
            int(round(p.Y / q)),
            int(round(p.Z / q)),
        )

    def __eq__(self, other):
        # Equal if the points fall into the same TOL-sized grid cell
        if isinstance(other, Node):
            return self._key_from_point(self.point) == self._key_from_point(other.point)
        if isinstance(other, geo.Point3d):
            return self._key_from_point(self.point) == self._key_from_point(other)
        return NotImplemented

    def __hash__(self):
        # Hash derived from the same quantized key used in __eq__
        return hash(self._key_from_point(self.point))


class Road:
    def __init__(self, region: geo.Curve, edge: Edge):
        self.region = region
        self.edge = edge


class Junction:
    def __init__(self, region, node: Node):
        self.region = region
        self.node = node
        self.connected_roads = []  # type: List[Road]


class RoadNetwork:
    def __init__(
        self,
        roads: List[Road],
        junctions: List[Junction],
        nodes: List[Node],
        edges: List[Edge],
    ):
        self.nodes = nodes
        self.edges = edges
        self.roads = roads
        self.junctions = junctions


class Block:
    def __init__(self, region: geo.Curve):
        self.region = region
        self.roads = []  # type: List[Road]
        self.junctions = []  # type: List[Junction]
