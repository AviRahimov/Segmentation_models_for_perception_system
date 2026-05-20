"""Real-time off-road robotics perception package.

Public sub-packages:

* ``perception.config``  — typed config schema + YAML loader
* ``perception.core``    — pure data contracts shared across modules
* ``perception.io``      — frame sources (video, camera, image directory)
* ``perception.models``  — instance + semantic models and inference backends
* ``perception.temporal``— causal smoothing (logit EMA, scene cuts, IoU tracker)
* ``perception.render``  — overlay primitives + display-mode-aware renderer
* ``perception.pipeline``— DI container that orchestrates per-frame inference
* ``perception.ui``      — PyQt5 video player (decoupled from inference)
* ``perception.datasets``— dataset downloaders (RUGD, ORFD)
"""

__version__ = "0.1.0"
