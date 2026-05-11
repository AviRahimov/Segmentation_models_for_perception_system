"""Vendored upstream model architectures (DDRNet, PP-LiteSeg).

Why vendor instead of pip-install
=================================

The two GOOSE-trained checkpoints we consume (DDRNet-39 and
PP-LiteSeg-B) come from the GOOSE benchmark training scripts at
``github.com/FraunhoferIOSB/goose_dataset``, which use **Deci-AI's
``super_gradients``** as their training framework. We do not pip-install
``super_gradients`` because it pulls in a few hundred MB of training /
data-pipeline / Hydra / experiment-tracking deps that are completely
irrelevant to inference, and it would impose a Torch version ceiling.

Files in this directory
-----------------------

* :mod:`ddrnet39_goose` — **THE** DDRNet architecture used by the
  comparison harness. Vendored from ``Deci-AI/super-gradients``
  (Apache-2.0); strict-loads the official GOOSE category checkpoint
  ``ddrnet_category_512.pth``. This is the file the wrapper at
  ``perception.models.semantic.ddrnet`` actually imports.

* :mod:`ddrnet23_slim` — earlier hypothesis (chenjun2hao DDRNet-23-slim,
  MIT). Kept on disk for reference / future rounds; **superseded** by
  ``ddrnet39_goose`` for the GOOSE-published weights, which require the
  ``super_gradients`` layer naming. Not imported by any production
  wrapper today.

* :mod:`ppliteseg` — PP-LiteSeg-B port (midasklr / PaddlePaddle
  Apache-2.0). Kept on disk per the project plan, but **not used in the
  current comparison round**: the GOOSE-published PP-LiteSeg checkpoint
  comes from the same ``super_gradients`` framework whose key layout
  differs structurally from this port, and PP-LiteSeg is dropped from
  this round's comparison anyway (DDRNet > PP-LiteSeg on the public
  GOOSE leaderboard for 12-class and 64-class splits, so the parent
  agent decided it isn't worth the load-path engineering).

Source headers
--------------

Each vendored file's top-of-file docstring lists::

    Source: <upstream repo URL>@<commit sha>
    License: <SPDX id>

How to update
-------------

To pull a fresh upstream copy, re-run ``curl`` against the raw GitHub
URL listed in each file's header, replace the file, and bump the
``commit sha``. Keep the diff to upstream small and clearly documented
in the file header so the next maintainer can re-apply our local edits
on top of a fresh upstream snapshot.

Public surface
--------------

The wrappers (``perception.models.semantic.ddrnet`` and
``perception.models.semantic.ppliteseg``) only import the top-level
builder they need (``ddrnet_39_goose``, ``ppliteseg_b``); everything
else in these files is implementation detail.
"""
