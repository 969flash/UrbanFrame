# -*- coding:utf-8 -*-
try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import road_net
import importlib, utils

importlib.reload(utils)
importlib.reload(road_net)


def snap_roads_pt(roads: List[Tuple[geo.Curve, float]], tol: float) -> List[geo.Curve]:
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


def snap_roads_crv(roads: List[Tuple[geo.Curve, float]], tol: float) -> List[geo.Curve]:
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


def snap_roads(road_data: List[Tuple[geo.Curve, float]], tol: float) -> List[geo.Curve]:
    """도로 네트워크의 모든 교차점을 스냅합니다."""
    if len(road_data) < 2:
        return road_data

    road_data = snap_roads_crv(road_data, tol)
    road_data = snap_roads_pt(road_data, tol)

    return road_data


if __name__ == "__main__":
    # inputs setting
    road_data = globals().get("road_data", [])
    road_data = snap_roads(road_data, tol=2)
    road_network = road_net.RoadNetwork()
    road_network.generate(road_data)

    edges = [e.curve for e in road_network.edges]
    nodes = [n.point for n in road_network.nodes]
    roads = [r.region for r in road_network.roads]
    junctions = [j.region for j in road_network.junctions]
    print(
        "RoadNetwork: {} edges, {} nodes, {} roads, {} junctions".format(
            len(edges), len(nodes), len(roads), len(junctions)
        )
    )
