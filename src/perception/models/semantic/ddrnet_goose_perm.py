"""GOOSE-12 channel alignment for ``ddrnet_category_512.pth``.

The published 12-way head does not list classes in the same order as
``GOOSE_12_NAMES`` / ``goose_label_mapping.csv``. Before softmax, we gather
logits so axis ``c`` matches that canonical ordering.

``DOC_SLOT_TO_RAW_CHANNEL[c]`` is the **raw checkpoint channel index** that
feeds **canonical** GOOSE class ``c`` (vegetation=0 … void=11).

Regenerate with::

    PYTHONPATH=src python scripts/calibrate_ddrnet_goose_channels.py \\
        --max-frames 200 --write-perm
"""
from __future__ import annotations

# Greedy match on GOOSE-Ex val (calibrate_ddrnet_goose_channels.py).
DOC_SLOT_TO_RAW_CHANNEL: tuple[int, ...] = (5, 8, 3, 4, 0, 2, 10, 7, 9, 1, 11, 6)
