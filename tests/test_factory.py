"""Tests for the model and source factories.

We cannot actually instantiate YOLOE / SegFormer (they require weight
downloads and a GPU), so we test the registry dispatch logic and the
clear-error path for unknown names.
"""
import pytest

from perception.config.schema import (
    HardwareCfg,
    InstanceModelCfg,
    SemanticModelCfg,
    SourceCfg,
)
from perception.io.factory import build_source
from perception.models import factory as model_factory
from perception.models.backends.factory import build_backend
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.backends.tensorrt import TensorRTBackend


def test_instance_registry_known_names():
    for n in ("yoloe26l", "yoloe-26l", "yoloe", "YoloE26L"):
        assert n.lower().strip() in model_factory.INSTANCE_REGISTRY


def test_unknown_instance_model_raises():
    with pytest.raises(ValueError, match="Unknown instance model"):
        model_factory.build_instance_model(
            InstanceModelCfg(name="not-a-model"),
            HardwareCfg(device="cpu", fp16=False),
            backend=PyTorchBackend(),
        )


def test_unknown_semantic_model_raises():
    with pytest.raises(ValueError, match="Unknown semantic model"):
        model_factory.build_semantic_model(
            SemanticModelCfg(name="bogus"),
            HardwareCfg(device="cpu", fp16=False),
            backend=PyTorchBackend(),
        )


def test_backend_factory_default_is_pytorch():
    be = build_backend(use_tensorrt=False)
    assert isinstance(be, PyTorchBackend)
    assert be.is_available()


def test_backend_factory_trt_unavailable_raises():
    # On any test machine that does not ship tensorrt, the factory must
    # raise instead of silently downgrading.
    if TensorRTBackend().is_available():
        pytest.skip("tensorrt is installed; cannot test the unavailable path")
    with pytest.raises(RuntimeError, match="tensorrt"):
        build_backend(use_tensorrt=True)


def test_source_factory_unknown_type_raises():
    with pytest.raises(ValueError):
        build_source(SourceCfg(type="ftp"))  # type: ignore[arg-type]


def test_source_factory_video_requires_path():
    with pytest.raises(ValueError, match="source.path"):
        build_source(SourceCfg(type="video", path=None))


def test_image_dir_factory_requires_path():
    with pytest.raises(ValueError, match="source.path"):
        build_source(SourceCfg(type="image_dir", path=None))
