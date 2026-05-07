"""Typed configuration schema and YAML loader."""
from .schema import (
    AppConfig,
    ClassDef,
    DatasetsCfg,
    HardwareCfg,
    InstanceModelCfg,
    InstanceSAM2Cfg,
    ModelsCfg,
    PlayerCfg,
    SemanticEMACfg,
    SemanticModelCfg,
    SourceCfg,
    TemporalCfg,
)
from .loader import load_config, override_source

__all__ = [
    "AppConfig",
    "ClassDef",
    "DatasetsCfg",
    "HardwareCfg",
    "InstanceModelCfg",
    "InstanceSAM2Cfg",
    "ModelsCfg",
    "PlayerCfg",
    "SemanticEMACfg",
    "SemanticModelCfg",
    "SourceCfg",
    "TemporalCfg",
    "load_config",
    "override_source",
]
