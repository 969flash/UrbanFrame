try:
    from typing import List, Tuple
except ImportError:
    pass
import networkx as nx
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp
import utils
import importlib
import scriptcontext as sc
import units

from units import Node, Edge, Road, Junction, RoadNetwork

importlib.reload(units)
importlib.reload(utils)
TOL = 5


class RoadNetworkGenerator:
    def __init__(self):
        pass

    def generate(self, road_data: List[Tuple[geo.Curve, float]]) -> RoadNetwork:
        """도로 네트워크를 생성합니다."""

        road_data = self.snap_roads(road_data, tol=2)
        nodes, edges = self._get_node_and_edge(road_data)
        roads, junctions = self._get_road_and_junction(nodes, edges)
        road_network = RoadNetwork(roads, junctions, nodes, edges)
        return road_network

    def snap_roads_pt(
        self, roads: List[Tuple[geo.Curve, float]], tol: float
    ) -> List[geo.Curve]:
        """도로 네트워크의 모든 교차점을 스냅합니다."""
        if len(roads) < 2:
            return roads

        # 모든 커브 쌍의 교차점 수집
        all_crvs = [road[0] for road in roads]
        pts = utils.get_pts_from_crvs(all_crvs)
        for crv in all_crvs:
            pts.extend(utils.get_vertices(crv))

        snaped_road_datas = []
        for road_crv, width in sorted(roads, key=lambda x: x[1]):
            for pt in pts:
                if pt.EpsilonEquals(road_crv.PointAtStart, tol):
                    road_crv = utils.move_curve_endpoint(road_crv, pt, "start")

                if pt.EpsilonEquals(road_crv.PointAtEnd, tol):
                    road_crv = utils.move_curve_endpoint(road_crv, pt, "end")
            snaped_road_datas.append((road_crv, width))

        return snaped_road_datas

    def snap_roads_crv(
        self, roads: List[Tuple[geo.Curve, float]], tol: float
    ) -> List[geo.Curve]:
        """도로 네트워크의 모든 교차점을 스냅합니다."""
        if len(roads) < 2:
            return roads

        snaped_road_datas = []
        for road_crv, width in sorted(roads, key=lambda x: x[1]):
            other_road_crvs = [crv for crv, w in roads if crv != road_crv]
            for other_road_crv in other_road_crvs:
                _, param = other_road_crv.ClosestPoint(road_crv.PointAtStart)
                point_at = other_road_crv.PointAt(param)
                if point_at.DistanceTo(road_crv.PointAtStart) <= tol:
                    road_crv = utils.move_curve_endpoint(road_crv, point_at, "start")

                _, param = other_road_crv.ClosestPoint(road_crv.PointAtEnd)
                point_at = other_road_crv.PointAt(param)
                if point_at.DistanceTo(road_crv.PointAtEnd) <= tol:
                    road_crv = utils.move_curve_endpoint(road_crv, point_at, "end")

            snaped_road_datas.append((road_crv, width))

        return snaped_road_datas

    def snap_roads(
        self, road_data: List[Tuple[geo.Curve, float]], tol: float
    ) -> List[geo.Curve]:
        """도로 네트워크의 모든 교차점을 스냅합니다."""
        if len(road_data) < 2:
            return road_data

        road_data = self.snap_roads_crv(road_data, tol)
        road_data = self.snap_roads_pt(road_data, tol)

        return road_data

    def _get_node_and_edge(
        self, road_data: List[Tuple[geo.Curve, float]]
    ) -> Tuple[List[Node], List[Edge]]:
        # 입력 정규화: 단일 커브 리스트 또는 (커브, 폭) 리스트 모두 허용
        # 모든 커브 쌍의 교차점 수집
        curves = [rd[0] for rd in road_data]
        node_pts = utils.get_pts_from_crvs(curves)
        edges = self._get_edges(road_data, node_pts)

        nodes = []
        for node_pt in node_pts:
            node = Node(node_pt)
            connected_edges = [e for e in edges if e.is_at(node)]
            if not connected_edges:
                raise ValueError(f"No connected edges found for node at {node_pt}")
            node.add_edges(connected_edges)
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

    def _get_road_and_junction(
        self, nodes: List[Node], edges: List[Edge]
    ) -> Tuple[List[Road], List[Junction]]:
        junctions = []  # type: List[Junction]
        for node in nodes:
            junction_region = self._get_junction_region(node)
            junction = Junction(junction_region, node)
            junctions.append(junction)

        roads = []  # type: List[Road]
        for edge in edges:
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

        try:
            diff_region = utils.get_difference_regions(
                road_region, junction_regions, 0.5
            )
        except:
            raise ValueError("도로크기에 비해 교차로가 너무 큽니다.")

        return max(diff_region, key=lambda r: r.GetLength())
