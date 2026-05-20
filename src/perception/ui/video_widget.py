"""Frame display widget."""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class VideoCanvas(QLabel):
    """Displays a BGR numpy frame, scaled with aspect ratio preserved."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background: #111;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self._last_frame: np.ndarray | None = None

    def show_frame_bgr(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        self._last_frame = frame
        h, w = frame.shape[:2]
        # Convert BGR -> RGB without unnecessary copies.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        if self._last_frame is not None:
            self.show_frame_bgr(self._last_frame)
