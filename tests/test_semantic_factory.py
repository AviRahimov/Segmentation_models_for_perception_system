"""Tests for the semantic-model factory: registry dispatch + default weights."""
import pytest

from perception.config.schema import HardwareCfg, SemanticModelCfg
from perception.models import factory as model_factory
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.semantic.auriganet import AurigaNetSemanticModel
from perception.models.semantic.segformer import SegFormerSemanticModel


_CPU_HW = HardwareCfg(device="cpu", fp16=False)


# --------------------------------------------------------------------------- #
# SegFormer                                                                    #
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
# AurigaNet                                                                    #
# --------------------------------------------------------------------------- #


def test_auriganet_dispatch_cpu():
    """Factory dispatches 'auriganet' → AurigaNetSemanticModel on CPU."""
    cfg = SemanticModelCfg(name="auriganet", weights="")
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert isinstance(m, AurigaNetSemanticModel)


def test_auriganet_num_classes_propagated():
    """num_classes kwarg reaches the model."""
    cfg = SemanticModelCfg(name="auriganet", weights="", num_classes=3)
    m = model_factory.build_semantic_model(cfg, _CPU_HW, backend=PyTorchBackend())
    assert m._num_classes == 3


# --------------------------------------------------------------------------- #
# Default-weights table sanity                                                 #
# --------------------------------------------------------------------------- #


def test_default_weights_table_covers_every_registry_key():
    missing = (
        set(model_factory.SEMANTIC_REGISTRY)
        - set(model_factory.SEMANTIC_DEFAULT_WEIGHTS)
    )
    assert not missing, f"missing default weights for: {sorted(missing)}"
