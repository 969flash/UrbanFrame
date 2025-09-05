try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp
import utils
import importlib

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


class RoadNetwork:
    def __init__(self):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

    def generate(self, road_data: List[Tuple[geo.Curve, float]]) -> None:
        # 입력 정규화: 단일 커브 리스트 또는 (커브, 폭) 리스트 모두 허용
        # 모든 커브 쌍의 교차점 수집
        curves = [rd[0] for rd in road_data]
        node_pts = utils.get_pts_from_crvs(curves)
        self.edges = self._get_edges(road_data, node_pts)
        for node_pt in node_pts:
            node = Node(node_pt)
            connected_edges = [e for e in self.edges if e.is_at(node)]
            node.add_edges(connected_edges)
            self.nodes.append(node)

    def _get_edges(
        self, road_data: List[Tuple[geo.Curve, float]], node_pts: List[geo.Point3d]
    ) -> List[Edge]:
        edges = []
        for curve, width in road_data:
            split_curves = utils.split_curve_at_pts(curve, node_pts)
            for sc in split_curves:
                edge = Edge(sc, width)
                edges.append(edge)
        self.edges = edges
        return edges


class Road:
    def __init__(self, region: geo.Curve, edge: Edge):
        self.region = region
        self.edge = edge


class Junction:
    def __init__(self, region, node: Node):
        self.region = region
        self.node = node
        self.connected_roads = []  # type: List[Road]


class RoadGenerator:
    def __init__(self):
        self.roads = []  # type: List[Road]
        self.junctions = []  # type: List[Junction]

    def generate(self, road_network: RoadNetwork) -> None:
        junctions = []  # type: List[Junction]
        for node in road_network.nodes:
            junction_region = self._get_junction_region(node)
            junction = Junction(junction_region, node)
            junctions.append(junction)

        roads = []  # type: List[Road]
        for edge in road_network.edges:
            road_region = self._get_road_region(edge, junctions)
            road = Road(road_region, edge)
            roads.append(road)

        for junction in junctions:
            connected_roads = [r for r in roads if r.edge.is_at(junction.node)]
            junction.connected_roads = connected_roads

        self.roads = roads
        self.junctions = junctions

        return self.roads, self.junctions

    def _get_junction_region(self, node: Node) -> geo.Curve:
        offset_crvs = []

        for edge in node.edges:
            offset_result = utils.offset_crv_outward(edge.curve, edge.width / 2)
            if not offset_result:
                raise ValueError("Failed to offset road curve.")
            offset_crvs.extend(offset_result)

        intersections = utils.get_pts_from_crvs(offset_crvs)
        if not intersections:
            raise ValueError("No intersection found for junction region.")

        return ghcomp.ConvexHull(intersections, geo.Plane.WorldXY).hull

    def _get_road_region(self, edge: Edge, junctions: List[Junction]) -> geo.Curve:
        start_junction = next(
            (j for j in junctions if j.node == edge.curve.PointAtStart), None
        )
        end_junction = next(
            (j for j in junctions if j.node == edge.curve.PointAtEnd), None
        )

        road_region = utils.offset_crv_outward(edge.curve, edge.width / 2)
        if len(road_region) != 1:
            raise ValueError("offset result wrong.")

        diff_region = utils.get_difference_regions(
            road_region, [start_junction.region, end_junction.region]
        )

        return max(diff_region, key=lambda r: r.GetLength())
