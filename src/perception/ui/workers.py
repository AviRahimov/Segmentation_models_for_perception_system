"""Background workers: decoder thread + inference thread.

The two workers communicate through a bounded :class:`queue.Queue`:

    Decoder  --(idx, frame)-->  [queue]  --(idx, frame)-->  Inference
                                                         |
                                                         v
                                              result_ready signal -> UI

The UI thread does *no* model work; it only paints the latest QImage and
forwards user actions (pause, seek, speed) back to the workers.
"""
from __future__ import annotations

import logging
import queue

from PyQt6.QtCore import QMutex, QThread, pyqtSignal

from ..io.source_base import FrameSource
from ..pipeline.perception import PerceptionPipeline

logger = logging.getLogger(__name__)


class DecoderWorker(QThread):
    """Reads frames from a :class:`FrameSource` at a controlled rate."""

    finished_source = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        source: FrameSource,
        frame_queue: "queue.Queue",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._queue = frame_queue
        self._running = True
        self._paused = False
        self._speed = 1.0
        self._seek_request: int | None = None
        self._mutex = QMutex()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            target_fps = max(1.0, self._source.fps())
            while self._running:
                if self._paused:
                    self.msleep(20)
                    continue

                self._mutex.lock()
                seek_to = self._seek_request
                self._seek_request = None
                self._mutex.unlock()
                if seek_to is not None:
                    self._source.seek(seek_to)
                    # Drain the queue so stale frames don't render after seek.
                    self._drain_queue()

                ok, frame = self._source.read()
                if not ok or frame is None:
                    self.finished_source.emit()
                    break
                idx = max(0, self._source.position - 1)

                # Bounded queue: if inference is behind, wait then retry.
                while self._running:
                    try:
                        self._queue.put((idx, frame), timeout=0.5)
                        break
                    except queue.Full:
                        if self._paused or self._seek_request is not None:
                            break

                if not self._paused:
                    step = 1.0 / (target_fps * max(0.1, self._speed))
                    self.msleep(max(0, int(step * 1000)))
        except Exception as e:  # noqa: BLE001
            logger.exception("DecoderWorker crashed")
            self.error.emit(str(e))

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._running = False

    def pause(self, on: bool) -> None:
        self._paused = bool(on)

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, float(speed))

    def request_seek(self, frame_idx: int) -> None:
        self._mutex.lock()
        self._seek_request = int(frame_idx)
        self._mutex.unlock()

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            return


class InferenceWorker(QThread):
    """Consumes (idx, frame) tuples and emits :class:`FrameResult`s."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        pipeline: PerceptionPipeline,
        frame_queue: "queue.Queue",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._queue = frame_queue
        self._running = True
        self._reset_request = False

    def run(self) -> None:
        try:
            while self._running:
                if self._reset_request:
                    self._pipeline.reset_temporal()
                    self._reset_request = False
                try:
                    idx, frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                result = self._pipeline.process(frame, idx)
                self.result_ready.emit(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("InferenceWorker crashed")
            self.error.emit(str(e))

    def stop(self) -> None:
        self._running = False

    def request_reset(self) -> None:
        self._reset_request = True
