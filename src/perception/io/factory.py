"""Factory that maps :class:`SourceCfg` -> concrete :class:`FrameSource`."""
from __future__ import annotations

from ..config.schema import SourceCfg
from .camera_source import CameraSource
from .image_dir_source import ImageDirSource
from .source_base import FrameSource
from .video_source import VideoFileSource


def build_source(cfg: SourceCfg) -> FrameSource:
    if cfg.type == "video":
        if not cfg.path:
            raise ValueError("source.path is required when source.type='video'")
        return VideoFileSource(cfg.path)
    if cfg.type == "camera":
        return CameraSource(cfg.camera_index, fps_hint=cfg.fps_hint)
    if cfg.type == "image_dir":
        if not cfg.path:
            raise ValueError("source.path is required when source.type='image_dir'")
        return ImageDirSource(cfg.path, glob=cfg.image_dir_glob, fps_hint=cfg.fps_hint)
    raise ValueError(f"Unknown source.type: {cfg.type!r}")
