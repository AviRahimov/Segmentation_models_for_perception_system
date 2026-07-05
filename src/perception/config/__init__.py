"""Typed configuration schema and YAML loader."""
from .schema import (
    AppConfig,
    ClassDef,
    HardwareCfg,
    InstanceModelCfg,
    InstancePromptMode,
    ModelsCfg,
    PlayerCfg,
    SemanticEMACfg,
    SemanticModelCfg,
    SourceCfg,
    TemporalCfg,
)
from .loader import load_config, override_source, resolve_path_relative_config

__all__ = [
    "AppConfig",
    "ClassDef",
    "HardwareCfg",
    "InstanceModelCfg",
    "InstancePromptMode",
    "ModelsCfg",
    "PlayerCfg",
    "SemanticEMACfg",
    "SemanticModelCfg",
    "SourceCfg",
    "TemporalCfg",
    "load_config",
    "override_source",
    "resolve_path_relative_config",
]
