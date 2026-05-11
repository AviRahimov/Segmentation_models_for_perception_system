"""Tests for the semantic-model factory: registry dispatch + default weights.

We never let the inner model classes touch the network / disk; they're
either monkeypatched (SegFormer's transformers loaders) or are
checkpoint-free skeletons in Stage 1 (DDRNet, PP-LiteSeg).
"""
import pytest

from perception.config.schema import HardwareCfg, SemanticModelCfg
from perception.models import factory as model_factory
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.semantic.ddrnet import DDRNetSemanticModel
from perception.models.semantic.ppliteseg import PPLiteSegSemanticModel
from perception.models.semantic.segformer import SegFormerSemanticModel


_CPU_HW = HardwareCfg(device="cpu", fp16=False)


# --------------------------------------------------------------------------- #
# SegFormer-B2                                                                 #
# --------------------------------------------------------------------------- #


class _FakeSegformerModel:
    class _Config:
        num_labels = 150

    config = _Config()

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def half(self):
        return self

    def parameters(self):  # pragma: no cover - never reached in these tests
        import torch

        yield torch.zeros(1)


class _FakeSegformerProcessor:
    pass


@pytest.fixture
def patched_segformer(monkeypatch):
    """Stub out HuggingFace ``from_pretrained`` so we never hit the network."""
    captured: dict[str, str] = {}

    def _fake_processor_from_pretrained(weights, *_a, **_k):
        captured["processor_weights"] = weights
        return _FakeSegformerProcessor()

    def _fake_model_from_pretrained(weights, *_a, **_k):
        captured["model_weights"] = weights
        return _FakeSegformerModel()

    import transformers  # type: ignore

    monkeypatch.setattr(
        transformers.SegformerImageProcessor,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _fake_processor_from_pretrained(*a, **k)),
    )
    monkeypatch.setattr(
        transformers.SegformerForSemanticSegmentation,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _fake_model_from_pretrained(*a, **k)),
    )
    return captured


def test_segformer_b2_dispatch_and_weights(patched_segformer):
    cfg = SemanticModelCfg(name="segformer-b2", weights="")
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert isinstance(m, SegFormerSemanticModel)
    assert patched_segformer["model_weights"] == (
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )


def test_segformer_b4_dispatch_and_weights(patched_segformer):
    cfg = SemanticModelCfg(name="segformer-b4", weights="")
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert isinstance(m, SegFormerSemanticModel)
    assert patched_segformer["model_weights"] == (
        "nvidia/segformer-b4-finetuned-ade-512-512"
    )


def test_segformer_explicit_weights_override(patched_segformer):
    cfg = SemanticModelCfg(name="segformer-b2", weights="my-org/my-segformer")
    model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert patched_segformer["model_weights"] == "my-org/my-segformer"


# --------------------------------------------------------------------------- #
# DDRNet (Stage 2: real ``ddrnet_39_goose`` build + strict weight load).
#
# We don't want unit tests to hit a ~250 MB checkpoint on disk, so we
# monkeypatch the architecture builder and ``torch.load`` to return
# fakes that share state-dict keys / shapes with the real model. The
# integration test that exercises the *actual* GOOSE weights lives in
# ``tests/test_ddrnet_wrapper.py`` and is skipped when the checkpoint
# file is absent.
# --------------------------------------------------------------------------- #


class _FakeDDRNetModule:
    """Minimal stand-in for the vendored DDRNet module.

    We don't bother building real weights — the factory test only needs
    ``load_state_dict`` / ``eval`` / ``to`` / ``half`` / ``parameters``
    to behave plausibly.
    """

    def __init__(self):
        import torch

        self._dummy = torch.zeros(1)

    def load_state_dict(self, sd, strict=True):  # noqa: ARG002 - mirror torch API
        return None

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def half(self):
        return self

    def parameters(self):
        yield self._dummy


@pytest.fixture
def patched_ddrnet(monkeypatch):
    """Stub the DDRNet builder + ``torch.load`` so tests are network/disk-free."""
    captured: dict[str, object] = {}

    def _fake_builder(num_classes, use_aux_heads=False):  # noqa: ARG001
        captured["num_classes"] = num_classes
        return _FakeDDRNetModule()

    def _fake_torch_load(path, *_a, **_k):
        captured["weights_path"] = path
        return {"net": {}, "acc": 0.0, "epoch": 0}

    import torch

    from perception.models.semantic import ddrnet as ddrnet_mod

    monkeypatch.setattr(ddrnet_mod, "ddrnet_39_goose", _fake_builder)
    monkeypatch.setattr(torch, "load", _fake_torch_load)
    return captured


def test_ddrnet_dispatch_and_weights(patched_ddrnet):
    cfg = SemanticModelCfg(name="ddrnet", weights="")
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert isinstance(m, DDRNetSemanticModel)
    # Default points at the local on-disk checkpoint, not a URL --
    # the wrapper does plain ``torch.load(path)``.
    assert m._weights == "weights/ddrnet_category_512.pth"
    assert patched_ddrnet["weights_path"] == "weights/ddrnet_category_512.pth"
    assert patched_ddrnet["num_classes"] == 12


def test_ddrnet_alias_dispatch(patched_ddrnet):
    for alias in ("ddrnet-39", "ddrnet39", "DDRNet"):
        cfg = SemanticModelCfg(name=alias, weights="")
        m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
        assert isinstance(m, DDRNetSemanticModel)


# --------------------------------------------------------------------------- #
# PP-LiteSeg (still a Stage-1 skeleton; constructor doesn't load weights)     #
# --------------------------------------------------------------------------- #


def test_ppliteseg_dispatch_and_weights():
    cfg = SemanticModelCfg(name="ppliteseg", weights="")
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert isinstance(m, PPLiteSegSemanticModel)
    # Skeleton points at the local download path (the on-disk file may
    # not actually exist; the constructor just records the path).
    assert m._weights == "weights/ppliteseg_category_512.pth"


def test_ppliteseg_alias_dispatch():
    for alias in ("pp-liteseg", "ppliteseg-b2", "PP-LiteSeg-B2"):
        cfg = SemanticModelCfg(name=alias, weights="")
        m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
        assert isinstance(m, PPLiteSegSemanticModel)


# --------------------------------------------------------------------------- #
# Default-weights table sanity                                                 #
# --------------------------------------------------------------------------- #


def test_default_weights_table_covers_every_registry_key():
    missing = (
        set(model_factory.SEMANTIC_REGISTRY)
        - set(model_factory.SEMANTIC_DEFAULT_WEIGHTS)
    )
    assert not missing, f"missing default weights for: {sorted(missing)}"
