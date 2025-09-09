try:
    from typing import List, Tuple
except ImportError:
    pass
import Rhino.Geometry as geo
import Rhino
import scriptcontext as sc
import System
import utils
import importlib

importlib.reload(utils)


generate_layer = globals().get("generate_layer", False)


def ensure_road_layers():
    """Switch to Rhino doc, create required layers, then restore ghdoc."""
    ghdoc = sc.doc  # Grasshopper doc (proxy)
    rhdoc = Rhino.RhinoDoc.ActiveDoc  # Real Rhino document
    # Always switch because GH proxy doc doesn't support Layers API fully
    sc.doc = rhdoc
    created = []

    def _ensure_layer(full_name):
        layers = rhdoc.Layers
        if layers.FindByFullPath(full_name, True) >= 0:
            return False
        parts = full_name.split("::")
        parent_id = System.Guid.Empty
        path_so_far = ""
        for part in parts:
            path_so_far = part if not path_so_far else path_so_far + "::" + part
            idx = layers.FindByFullPath(path_so_far, True)
            if idx >= 0:
                parent_id = layers[idx].Id
                continue
            layer = Rhino.DocObjects.Layer()
            layer.Name = part
            if parent_id != System.Guid.Empty:
                layer.ParentLayerId = parent_id
            new_index = layers.Add(layer)
            if new_index < 0:
                return False
            parent_id = layers[new_index].Id
        return True

    try:
        for name in [
            "Road Centerline",
            "Road Centerline::5m",
            "Road Centerline::10m",
        ]:
            if _ensure_layer(name):
                created.append(name)
    finally:
        # Restore Grasshopper document context
        sc.doc = ghdoc
    return created


def get_road_data_from_layer(base_layer_name="Road Centerline"):
    """Collect (curve, width) tuples from sublayers named like '<width>m'.

    Args:
        base_layer_name: Top-level layer containing width sublayers (e.g. 'Road Centerline').
    Returns:
        List[Tuple[geo.Curve, float]] suitable for RoadNetworkGenerator.generate.
    """
    ghdoc = sc.doc
    rhdoc = Rhino.RhinoDoc.ActiveDoc
    sc.doc = rhdoc
    try:
        import re

        layers = rhdoc.Layers
        root_index = layers.FindByFullPath(base_layer_name, True)
        if root_index < 0:
            return []
        root_id = layers[root_index].Id

        pattern = re.compile(r"^(?P<val>\d+(?:\.\d+)?)m$")
        width_layer_indices = {}
        for i in range(layers.Count):
            layer = layers[i]
            if layer is None or not layer.IsValid:
                continue
            if layer.ParentLayerId != root_id:
                continue
            name = (layer.Name or "").strip()
            m = pattern.match(name)
            if not m:
                continue
            try:
                width_layer_indices[i] = float(m.group("val"))
            except:  # noqa: E722
                continue

        if not width_layer_indices:
            return []

        road_data = []
        # Single pass over objects
        for obj in rhdoc.Objects:
            attr = obj.Attributes
            idx = attr.LayerIndex
            width = width_layer_indices.get(idx)
            if width is None:
                continue
            geo_obj = obj.Geometry

            if not isinstance(geo_obj, geo.Curve):
                continue
            try:
                segs = utils.explode_curve(geo_obj)
                for seg in segs:
                    road_data.append((seg.DuplicateCurve(), width))
            except Exception:
                continue
        return road_data
    finally:
        sc.doc = ghdoc
