# r: python-louvain
try:
    from typing import List, Tuple
except ImportError:
    pass
import networkx as nx
import community as community_louvain
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp
import utils
import importlib
import scriptcontext as sc
import units

from units import Node, Edge, Road, Junction, RoadNetwork


class RoadNetworkAnalyzer:
    def __init__(self, road_network: RoadNetwork):
        self.road_network = road_network
        self.graph = road_network.graph

    def total_road_length(self) -> float:
        """Calculate the total length of all roads in the network."""
        return sum(edge.curve.GetLength() for edge in self.road_network.edges)

    def average_road_width(self) -> float:
        """Calculate the average width of all roads in the network."""
        if not self.road_network.edges:
            return 0.0
        return sum(edge.width for edge in self.road_network.edges) / len(
            self.road_network.edges
        )

    def node_degree_distribution(self) -> dict:
        """Return a dictionary with node degrees as keys and their frequencies as values."""
        degree_count = {}
        for node in self.road_network.nodes:
            degree = len(node.edges)
            if degree not in degree_count:
                degree_count[degree] = 0
            degree_count[degree] += 1
        return degree_count

    def betweenness_centrality(self) -> dict:
        """Calculate the betweenness centrality for each node in the network."""
        return nx.betweenness_centrality(self.graph)

    def closeness_centrality(self) -> dict:
        """Calculate the closeness centrality for each node in the network."""
        return nx.closeness_centrality(self.graph)

    def eighenvector_centrality(self) -> dict:
        """Calculate the eigenvector centrality for each node in the network."""
        return nx.eigenvector_centrality(self.graph, max_iter=1000)

    def detect_communities(self) -> dict:
        """Detect communities in the road network using the Louvain method."""
        return community_louvain.best_partition(self.graph)


class ResultVisualizer:
    """Create color-coded preview data for road network analysis results.

    Each visualize_* method returns a tuple: (curves, colors, widths)
      - curves: list[geo.Curve]
      - colors: list[System.Drawing.Color] (same length)
      - widths: list[int] (constant 3 for preview thickness)
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        start_color=(0, 255, 0),  # green
        end_color=(255, 0, 0),  # red
    ):
        self.road_network = road_network
        self.graph = road_network.graph
        self.start_color = start_color
        self.end_color = end_color

    # ------------- internal helpers ------------- #
    @staticmethod
    def _interpolate_color(start_rgb, end_rgb, t: float):
        import System

        t = max(0.0, min(1.0, t))
        r = int(round(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t))
        g = int(round(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t))
        b = int(round(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t))
        return System.Drawing.Color.FromArgb(255, r, g, b)

    @staticmethod
    def _normalize(values):
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-12:
            return [0.5] * len(values)
        k = 1.0 / (vmax - vmin)
        return [(v - vmin) * k for v in values]

    def _edges_with_metric(self, metric_dict):
        # Builds parallel lists of curves and metric values
        curves = []
        values = []
        # Edge mapping: we have Road objects referencing Edge w/ curve
        # We'll iterate through road_network.edges (Edge objects) and try to locate metric.
        # metric_dict keys are node-based (for centralities) or node indexes; so we map via graph edges.
        if not metric_dict:
            return curves, values
        for u, v, data in self.graph.edges(data=True):
            curve = data.get("curve")
            if not curve:
                # fallback attempt: find matching edge in road_network.edges
                pass
            # For node-based metrics (centrality) we might have per-node values
            # For edge-based we expect a (u,v) key
            key_edge = metric_dict.get((u, v))
            if key_edge is None:
                key_edge = metric_dict.get((v, u))
            if key_edge is None:
                # attempt convert node metrics to edge by averaging
                if u in metric_dict and v in metric_dict:
                    key_edge = 0.5 * (metric_dict[u] + metric_dict[v])
            if key_edge is None:
                continue
            if curve is None:
                continue
            curves.append(curve)
            values.append(float(key_edge))
        return curves, values

    def _build(self, metric_dict):
        curves, values = self._edges_with_metric(metric_dict)
        if not curves:
            return [], [], []
        norm = self._normalize(values)
        colors = [
            self._interpolate_color(self.start_color, self.end_color, t) for t in norm
        ]
        widths = [3] * len(curves)
        return curves, colors, widths

    # ------------- public visualize methods ------------- #
    def visualize_betweenness(self, betweenness: dict):
        return self._build(betweenness)

    def visualize_communities(self, communities: dict):
        # communities: node -> community_id ; map to sequential ids then assign color scale
        if not communities:
            return [], [], []
        # Convert community id to numeric range
        comm_ids = list(set(communities.values()))
        comm_ids.sort()
        id_to_val = {cid: i for i, cid in enumerate(comm_ids)}
        # Build a node->normalized value dict
        max_val = float(len(comm_ids) - 1) if len(comm_ids) > 1 else 1.0
        node_metric = {
            n: (id_to_val[c] / max_val if max_val > 0 else 0.5)
            for n, c in communities.items()
        }
        return self._build(node_metric)

    def visualize_node_degrees(self, degree_distribution: dict):
        # degree_distribution: degree -> frequency; we need per-node degrees instead.
        # Recompute per-node degrees from graph then map to metric dict (node->degree)
        node_deg = dict(self.graph.degree())
        return self._build(node_deg)


road_network = globals().get("road_network", None)  # type: RoadNetwork

RoadNetworkAnalyzer(road_network)
rna = RoadNetworkAnalyzer(road_network)
betweenness = rna.betweenness_centrality()
vis = ResultVisualizer(road_network)
curves, colors, widths = vis.visualize_betweenness(betweenness)
