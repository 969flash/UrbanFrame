try:
    from typing import List, Tuple
except ImportError:
    pass
import units
import utils
import importlib
import Rhino.Geometry as geo
import scriptcontext as sc

importlib.reload(utils)
importlib.reload(units)
from units import Block, BlockInfo


# Minimal imports only
blocks = globals().get("blocks", [])  # type: List[Block]
block_infos = globals().get("block_infos", {})  # type: List[BlockInfo]


for block_info in block_infos:
    for block in blocks:
        if block.block_id in block_info.block_ids:
            block.initialize(block_info)
