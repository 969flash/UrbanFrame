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
    def __init__(self):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]
        self.roads = []  # type: List[Road]
        self.junctions = []  # type: List[Junction]

    def generate(self, road_data: List[Tuple[geo.Curve, float]]) -> None:
        self.nodes, self.edges = self._get_node_and_edge(road_data)
        self.roads, self.junctions = self._get_road_and_junction()

    def _get_node_and_edge(
        self, road_data: List[Tuple[geo.Curve, float]]
    ) -> Tuple[List[Node], List[Edge]]:
        # 입력 정규화: 단일 커브 리스트 또는 (커브, 폭) 리스트 모두 허용
        # 모든 커브 쌍의 교차점 수집
        curves = [rd[0] for rd in road_data]
        node_pts = utils.get_pts_from_crvs(curves)
        edges = self._get_edges(road_data, node_pts)
        print("EDGE COUNT:", len(edges))

        nodes = []
        for node_pt in node_pts:
            node = Node(node_pt)
            connected_edges = [e for e in edges if e.is_at(node)]
            node.add_edges(connected_edges)
            print("EDGE COOUNT:", len(connected_edges))
            nodes.append(node)

        return nodes, edges

    def _get_edges(
        self, road_data: List[Tuple[geo.Curve, float]], node_pts: List[geo.Point3d]
    ) -> List[Edge]:
        edges = []
        for curve, width in road_data:
            pts_on_crv = [pt for pt in node_pts if utils.is_pt_on_crv(pt, curve)]

            split_curves = utils.split_curve_at_pts(curve, pts_on_crv)
            for split_curve in split_curves:
                edge = Edge(split_curve, width)
                edges.append(edge)
        return edges

    def _get_road_and_junction(self) -> Tuple[List[Road], List[Junction]]:
        junctions = []  # type: List[Junction]
        for node in self.nodes:
            junction_region = self._get_junction_region(node)
            junction = Junction(junction_region, node)
            junctions.append(junction)

        roads = []  # type: List[Road]
        for edge in self.edges:
            road_region = self._get_road_region(edge, junctions)
            road = Road(road_region, edge)
            roads.append(road)

        for junction in junctions:
            connected_roads = [r for r in roads if r.edge.is_at(junction.node)]
            junction.connected_roads = connected_roads

        return roads, junctions

    def _get_junction_region(self, node: Node) -> geo.Curve:

        node_circle = geo.Circle(
            node.point, max(e.width for e in node.edges)
        ).ToNurbsCurve()

        intersections = []
        for edge in node.edges:
            offset_result = utils.offset_crv_outward(edge.curve, edge.width / 2)
            if not offset_result:
                raise ValueError("Failed to offset road curve.")
            for offset_crv in offset_result:
                inter_pts = utils.get_pts_from_crv_crv(node_circle, offset_crv)
                intersections.extend(inter_pts)

        if not intersections:
            raise ValueError("No intersection found for junction region.")

        result = ghcomp.ConvexHull(intersections, geo.Plane.WorldXY).hull

        return result

    def _get_road_region(self, edge: Edge, junctions: List[Junction]) -> geo.Curve:
        junction_regions = []
        # edge의 양 끝점에 있는 정션을 수집한다
        for j in junctions:
            if j.node == edge.curve.PointAtStart or j.node == edge.curve.PointAtEnd:
                junction_regions.append(j.region)

        road_region = utils.offset_crv_outward(edge.curve, edge.width / 2)
        if len(road_region) != 1:
            raise ValueError("offset result wrong.")

        if not junction_regions:
            return road_region[0]

        print(road_region, junction_regions)
        diff_region = utils.get_difference_regions(road_region, junction_regions, 0.5)

        return max(diff_region, key=lambda r: r.GetLength())
