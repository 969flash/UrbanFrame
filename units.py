try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp
import utils
import importlib
import networkx as nx
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

    @staticmethod
    def point_to_key(p: geo.Point3d):
        """Public: Quantize a point to TOL-sized grid for stable equality/hash."""
        q = TOL
        return (
            int(round(p.X / q)),
            int(round(p.Y / q)),
            int(round(p.Z / q)),
        )

    @property
    def point_key(self):
        """Public: Quantized key of this node's point."""
        return Node.point_to_key(self.point)

    def __eq__(self, other):
        # Equal if the points fall into the same TOL-sized grid cell
        if isinstance(other, Node):
            return Node.point_to_key(self.point) == Node.point_to_key(other.point)
        if isinstance(other, geo.Point3d):
            return Node.point_to_key(self.point) == Node.point_to_key(other)
        return NotImplemented

    def __hash__(self):
        # Hash derived from the same quantized key used in __eq__
        return hash(Node.point_to_key(self.point))


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

        self.graph = self._get_graph()

    def _get_graph(self):
        G = nx.Graph()

        # TOL 기반 좌표 양자화 키: Node.point_to_key / Node.point_key 재사용
        key = Node.point_to_key
        nodes_by_key = {n.point_key: n for n in self.nodes}

        # 노드 추가 + 속성
        for n in self.nodes:
            G.add_node(n, point=n.point, degree=len(n.edges))

        # 엣지 추가: 키 매핑 + 안전한 fallback
        for e in self.edges:
            n1 = nodes_by_key.get(key(e.curve.PointAtStart))
            n2 = nodes_by_key.get(key(e.curve.PointAtEnd))
            if not n1:
                n1 = next((n for n in self.nodes if n == e.curve.PointAtStart), None)
            if not n2:
                n2 = next((n for n in self.nodes if n == e.curve.PointAtEnd), None)
            if not n1 or not n2:
                continue
            G.add_edge(
                n1,
                n2,
                object=e,
                width=getattr(e, "width", None),
                length=e.curve.GetLength() if e.curve else None,
                curve=e.curve,
            )
        return G


class Block:
    def __init__(self, region: geo.Curve):
        self.region = region
        self.roads = []  # type: List[Road]
        self.junctions = []  # type: List[Junction]
