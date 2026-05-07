"""Main video player window."""
from __future__ import annotations

import logging
import queue
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from ..config.schema import AppConfig
from ..core.types import FrameResult
from ..io.source_base import FrameSource
from ..pipeline.perception import PerceptionPipeline
from ..render.renderer import Renderer
from .controls import PlayerControls
from .video_widget import VideoCanvas
from .workers import DecoderWorker, InferenceWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Glues source + pipeline + renderer together behind a Qt UI.

    The window itself owns no domain logic; both worker threads receive
    their dependencies via constructor injection.
    """

    QUEUE_SIZE = 4

    def __init__(
        self,
        source: FrameSource,
        pipeline: PerceptionPipeline,
        renderer: Renderer,
        config: AppConfig,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Off-Road Perception Player")
        self._source = source
        self._pipeline = pipeline
        self._renderer = renderer
        self._cfg = config

        # Layout ---------------------------------------------------------
        central = QWidget(self)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._canvas = VideoCanvas(self)
        v.addWidget(self._canvas, 1)
        self._controls = PlayerControls(
            total_frames=source.total_frames(),
            fps=source.fps(),
            default_speed=config.player.default_speed,
        )
        v.addWidget(self._controls)
        self.setCentralWidget(central)
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Workers --------------------------------------------------------
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._decoder = DecoderWorker(source, self._frame_queue)
        self._inferencer = InferenceWorker(pipeline, self._frame_queue)

        self._inferencer.result_ready.connect(self._on_result)
        self._decoder.error.connect(self._on_error)
        self._inferencer.error.connect(self._on_error)
        self._decoder.finished_source.connect(self._on_source_finished)

        self._controls.play_toggle.connect(self._on_pause)
        self._controls.seek_changed.connect(self._on_seek)
        self._controls.speed_changed.connect(self._decoder.set_speed)
        self._decoder.set_speed(config.player.default_speed)

        # Shortcuts ------------------------------------------------------
        self._wire_shortcuts()

        # FPS smoothing --------------------------------------------------
        self._smoothed_fps = 0.0
        self._fps_alpha = 0.1
        self._last_emit = time.perf_counter()

        # Start workers --------------------------------------------------
        self._decoder.start()
        self._inferencer.start()

    # ------------------------------------------------------------------ #
    def _wire_shortcuts(self) -> None:
        fps_int = max(1, int(round(self._source.fps())))

        def add(seq: str, fn) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)

        add("Space", self._toggle_play)
        add("F",     self._toggle_fullscreen)
        add("Q",     self.close)
        add("Esc",   self._exit_fullscreen)
        add("Right", lambda: self._seek_relative(+fps_int))
        add("Left",  lambda: self._seek_relative(-fps_int))
        add("Shift+Right", lambda: self._seek_relative(+fps_int * 10))
        add("Shift+Left",  lambda: self._seek_relative(-fps_int * 10))
        add("+", lambda: self._adjust_speed(+0.25))
        add("=", lambda: self._adjust_speed(+0.25))   # convenience without Shift
        add("-", lambda: self._adjust_speed(-0.25))

    # ------------------------------------------------------------------ #
    def closeEvent(self, e):  # noqa: N802 (Qt signature)
        self._decoder.stop()
        self._inferencer.stop()
        self._decoder.wait(2000)
        self._inferencer.wait(2000)
        try:
            self._source.release()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(e)

    # ------------------------------------------------------------------ #
    # Worker callbacks                                                    #
    # ------------------------------------------------------------------ #
    def _on_result(self, result: FrameResult) -> None:
        now = time.perf_counter()
        dt = max(1e-6, now - self._last_emit)
        self._last_emit = now
        inst_fps = 1.0 / dt
        self._smoothed_fps = (
            inst_fps if self._smoothed_fps == 0.0
            else (1.0 - self._fps_alpha) * self._smoothed_fps + self._fps_alpha * inst_fps
        )
        rendered = self._renderer.render(result, fps=self._smoothed_fps)
        self._canvas.show_frame_bgr(rendered)
        if self._source.total_frames() > 0:
            self._controls.update_position(result.frame_idx)
        self._status.showMessage(
            f"Frame {result.frame_idx}   "
            f"inference {result.inference_ms:.1f} ms   "
            f"{self._smoothed_fps:.1f} FPS   "
            + ("[scene cut]" if result.scene_cut else "")
        )

    def _on_error(self, msg: str) -> None:
        logger.error("Worker error: %s", msg)
        self._status.showMessage(f"ERROR: {msg}")

    def _on_source_finished(self) -> None:
        self._status.showMessage("End of stream")

    # ------------------------------------------------------------------ #
    # User controls                                                       #
    # ------------------------------------------------------------------ #
    def _on_pause(self, paused: bool) -> None:
        self._decoder.pause(paused)

    def _toggle_play(self) -> None:
        self._controls.btn_play.toggle()

    def _on_seek(self, frame_idx: int) -> None:
        self._decoder.request_seek(frame_idx)
        # Temporal context is broken across a seek -> drop EMA + tracker state.
        self._inferencer.request_reset()

    def _seek_relative(self, delta: int) -> None:
        if self._source.total_frames() <= 0:
            return
        new_idx = max(
            0,
            min(self._source.total_frames() - 1, self._source.position + delta),
        )
        self._controls.update_position(new_idx)
        self._on_seek(new_idx)

    def _adjust_speed(self, delta: float) -> None:
        new = max(0.1, self._controls.spin_speed.value() + delta)
        self._controls.spin_speed.setValue(new)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
