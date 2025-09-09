# -*- coding:utf-8 -*-
# Example landuse data dict for 7 blocks with 3 types (1-3, 4-5, 6-7)

example_landuse_data = {
    "type_A": {
        "block_ids": [1, 2, 3],
        "gfa": 5.0,  # 용적률
        "bcr": 0.45,  # 건폐율
        "landuses": ["residential", "retail"],
        "pedestrian_width": 3.0,
    },
    "type_B": {
        "block_ids": [4, 5],
        "gfa": 7.5,
        "bcr": 0.55,
        "landuses": ["office", "retail"],
        "pedestrian_width": 4.0,
    },
    "type_C": {
        "block_ids": [6, 7],
        "gfa": 3.5,
        "bcr": 0.35,
        "landuses": ["park", "community"],
        "pedestrian_width": 2.0,
    },
}

# Alternative per-block format also supported by landuse_setting.py
# example_landuse_data = {
#     1: {"gfa": 5.0, "bcr": 0.45, "landuses": ["residential", "retail"], "pedestrian_width": 3.0},
#     2: {"gfa": 5.0, "bcr": 0.45, "landuses": ["residential", "retail"], "pedestrian_width": 3.0},
#     3: {"gfa": 5.0, "bcr": 0.45, "landuses": ["residential", "retail"], "pedestrian_width": 3.0},
#     4: {"gfa": 7.5, "bcr": 0.55, "landuses": ["office", "retail"], "pedestrian_width": 4.0},
#     5: {"gfa": 7.5, "bcr": 0.55, "landuses": ["office", "retail"], "pedestrian_width": 4.0},
#     6: {"gfa": 3.5, "bcr": 0.35, "landuses": ["park", "community"], "pedestrian_width": 2.0},
#     7: {"gfa": 3.5, "bcr": 0.35, "landuses": ["park", "community"], "pedestrian_width": 2.0},
# }
