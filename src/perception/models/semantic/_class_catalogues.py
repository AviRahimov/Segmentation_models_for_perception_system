"""Native class catalogues for closed-vocabulary semantic models.

This module is deliberately data-only — no torch/transformers imports — so
that lightweight tests (and the compare-harness CLI) can pull the
canonical name lists without paying for a heavy import chain.

Catalogues exposed:

* :data:`ADE20K_150_NAMES` — the SceneParse150 ordering used by the
  CSAILVision sceneparsing benchmark and by all
  ``nvidia/segformer-*-finetuned-ade-*`` checkpoints. Index 0 is
  ``"wall"``, index 6 is ``"road"``, index 9 is ``"grass"``,
  index 46 is ``"sand"``, etc.

* :data:`GOOSE_12_NAMES` — the 12 "category"-level labels published by
  the GOOSE dataset (Mortimer et al., 2023 — see arXiv 2310.16788
  Table III). Ordering is the one we standardised across the project:
  vegetation, terrain, vehicle, object, construction, road, sign,
  human, sky, water, animal, void.

GOOSE-64 (the fine-grained label set) is intentionally **not** included;
this round only consumes GOOSE-12 checkpoints. Add ``GOOSE_64_NAMES``
in a follow-up if/when a 64-class checkpoint enters the pipeline.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ADE20K (SceneParse150) — canonical ordering. Index = native channel.
# Source: https://github.com/CSAILVision/sceneparsing
# ---------------------------------------------------------------------------
ADE20K_150_NAMES: tuple[str, ...] = (
    "wall",            # 0
    "building",        # 1
    "sky",             # 2
    "floor",           # 3
    "tree",            # 4
    "ceiling",         # 5
    "road",            # 6
    "bed",             # 7
    "windowpane",      # 8
    "grass",           # 9
    "cabinet",         # 10
    "sidewalk",        # 11
    "person",          # 12
    "earth",           # 13
    "door",            # 14
    "table",           # 15
    "mountain",        # 16
    "plant",           # 17
    "curtain",         # 18
    "chair",           # 19
    "car",             # 20
    "water",           # 21
    "painting",        # 22
    "sofa",            # 23
    "shelf",           # 24
    "house",           # 25
    "sea",             # 26
    "mirror",          # 27
    "rug",             # 28
    "field",           # 29
    "armchair",        # 30
    "seat",            # 31
    "fence",           # 32
    "desk",            # 33
    "rock",            # 34
    "wardrobe",        # 35
    "lamp",            # 36
    "bathtub",         # 37
    "railing",         # 38
    "cushion",         # 39
    "base",            # 40
    "box",             # 41
    "column",          # 42
    "signboard",       # 43
    "chest of drawers",# 44
    "counter",         # 45
    "sand",            # 46
    "sink",            # 47
    "skyscraper",      # 48
    "fireplace",       # 49
    "refrigerator",    # 50
    "grandstand",      # 51
    "path",            # 52
    "stairs",          # 53
    "runway",          # 54
    "case",            # 55
    "pool table",      # 56
    "pillow",          # 57
    "screen door",     # 58
    "stairway",        # 59
    "river",           # 60
    "bridge",          # 61
    "bookcase",        # 62
    "blind",           # 63
    "coffee table",    # 64
    "toilet",          # 65
    "flower",          # 66
    "book",            # 67
    "hill",            # 68
    "bench",           # 69
    "countertop",      # 70
    "stove",           # 71
    "palm",            # 72
    "kitchen island",  # 73
    "computer",        # 74
    "swivel chair",    # 75
    "boat",            # 76
    "bar",             # 77
    "arcade machine",  # 78
    "hovel",           # 79
    "bus",             # 80
    "towel",           # 81
    "light",           # 82
    "truck",           # 83
    "tower",           # 84
    "chandelier",      # 85
    "awning",          # 86
    "streetlight",     # 87
    "booth",           # 88
    "television",      # 89
    "airplane",        # 90
    "dirt track",      # 91
    "apparel",         # 92
    "pole",            # 93
    "land",            # 94
    "bannister",       # 95
    "escalator",       # 96
    "ottoman",         # 97
    "bottle",          # 98
    "buffet",          # 99
    "poster",          # 100
    "stage",           # 101
    "van",             # 102
    "ship",            # 103
    "fountain",        # 104
    "conveyer belt",   # 105
    "canopy",          # 106
    "washer",          # 107
    "plaything",       # 108
    "swimming pool",   # 109
    "stool",           # 110
    "barrel",          # 111
    "basket",          # 112
    "waterfall",       # 113
    "tent",            # 114
    "bag",             # 115
    "minibike",        # 116
    "cradle",          # 117
    "oven",            # 118
    "ball",            # 119
    "food",            # 120
    "step",            # 121
    "tank",            # 122
    "trade name",      # 123
    "microwave",       # 124
    "pot",             # 125
    "animal",          # 126
    "bicycle",         # 127
    "lake",            # 128
    "dishwasher",      # 129
    "screen",          # 130
    "blanket",         # 131
    "sculpture",       # 132
    "hood",            # 133
    "sconce",          # 134
    "vase",            # 135
    "traffic light",   # 136
    "tray",            # 137
    "ashcan",          # 138
    "fan",             # 139
    "pier",            # 140
    "crt screen",      # 141
    "plate",           # 142
    "monitor",         # 143
    "bulletin board",  # 144
    "shower",          # 145
    "radiator",        # 146
    "glass",           # 147
    "clock",           # 148
    "flag",            # 149
)
assert len(ADE20K_150_NAMES) == 150, "ADE20K-150 name list must have exactly 150 entries"


# ---------------------------------------------------------------------------
# GOOSE-12 (category level). Index = native channel.
# Ordering taken from arXiv 2310.16788 Table III.
# ---------------------------------------------------------------------------
GOOSE_12_NAMES: tuple[str, ...] = (
    "vegetation",   # 0
    "terrain",      # 1
    "vehicle",      # 2
    "object",       # 3
    "construction", # 4
    "road",         # 5
    "sign",         # 6
    "human",        # 7
    "sky",          # 8
    "water",        # 9
    "animal",       # 10
    "void",         # 11
)
assert len(GOOSE_12_NAMES) == 12, "GOOSE-12 name list must have exactly 12 entries"


CATALOGUE_SIZES: dict[str, int] = {
    "ade20k": len(ADE20K_150_NAMES),
    "goose_12": len(GOOSE_12_NAMES),
}
"""Per-catalogue native channel counts, keyed by the same allowlisted
``native_indices`` keys used in ``config.yaml``."""
