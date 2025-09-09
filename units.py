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
NODE_TOL = 0.5


class Edge:
    def __init__(self, curve: geo.Curve, width: float):
        self.curve = curve  # type: geo.Curve
        self.width = width  # type: float

    def is_at(self, node: "Node") -> bool:
        return node == self.curve.PointAtStart or node == self.curve.PointAtEnd


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
        # Allow positional tolerance of NODE_TOL
        if isinstance(other, Node):
            return self.point.DistanceTo(other.point) <= NODE_TOL
        if isinstance(other, geo.Point3d):
            return self.point.DistanceTo(other) <= NODE_TOL
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


class BlockInfo:
    """Configuration holder for initializing Block objects.

    Attributes:
        block_ids: List of target block ids this config applies to.
        gfa: Floor area ratio (용적률).
        bcr: Building coverage ratio (건폐율).
        landuses: List of landuse labels.
        pedestrian_width: Width of pedestrian zone (m).
    """

    def __init__(
        self,
        block_ids,  # type: List[int]
        gfa=None,  # type: float
        bcr=None,  # type: float
        landuses=None,  # type: List[str]
        pedestrian_width=None,  # type: float
    ):
        self.block_ids = block_ids
        self.gfa = gfa
        self.bcr = bcr
        self.landuses = landuses
        self.pedestrian_width = pedestrian_width


class Block:
    def __init__(self, region: geo.Curve, block_id: int = None):
        self.block_id = block_id
        self.region = region
        self.buildable_region = region  # type: geo.Curve
        self.roads = []  # type: List[Road]
        self.junctions = []  # type: List[Junction]

        # landuse/building defaults
        self.landuses = []  # type: List[str]
        self.gfa = 0.0  # 용적률
        self.bcr = 0.0  # 건폐율
        self.pedestrian_width = 0.0

        # placeholders for downstream
        self.pedestrian = []  # type: List[geo.Surface]
        self.buildings = []  # type: List[Building]

    def initialize(self, block_info: BlockInfo):
        """block_info의 설정으로 블록 초기화."""
        # gfa and bcr
        self.gfa = block_info.gfa
        self.bcr = block_info.bcr

        # landuses
        self.landuses = block_info.landuses

        # pedestrian width
        self.pedestrian_width = block_info.pedestrian_width

        self.buildable_region = utils.offset_regions_inward(
            self.region, self.pedestrian_width
        )[0]

        self.pedestrian = ghcomp.BoundarySurfaces([self.buildable_region, self.region])
