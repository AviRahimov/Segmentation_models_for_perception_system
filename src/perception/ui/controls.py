"""Player controls: play/pause, seek slider, position readout, speed."""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class PlayerControls(QWidget):
    play_toggle = pyqtSignal(bool)        # True == paused
    seek_changed = pyqtSignal(int)        # absolute frame index
    speed_changed = pyqtSignal(float)

    def __init__(
        self,
        total_frames: int,
        fps: float,
        default_speed: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fps = max(1.0, float(fps))
        self._total = max(1, int(total_frames)) if total_frames > 0 else 1
        self._has_total = total_frames > 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        self.btn_play = QPushButton("Pause")
        self.btn_play.setCheckable(True)
        self.btn_play.setChecked(False)
        self.btn_play.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self._total - 1))
        self.slider.setEnabled(self._has_total)
        self.slider.sliderReleased.connect(
            lambda: self.seek_changed.emit(int(self.slider.value()))
        )
        layout.addWidget(self.slider, 1)

        self.lbl_pos = QLabel(self._fmt(0))
        self.lbl_pos.setMinimumWidth(150)
        layout.addWidget(self.lbl_pos)

        layout.addWidget(QLabel("Speed:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8.0)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setValue(float(default_speed))
        self.spin_speed.valueChanged.connect(self.speed_changed.emit)
        layout.addWidget(self.spin_speed)

    # ------------------------------------------------------------------ #
    def _fmt(self, idx: int) -> str:
        sec = idx / self._fps
        m, s = divmod(int(sec), 60)
        if self._has_total:
            return f"{idx}/{self._total - 1}  {m:02d}:{s:02d}"
        return f"{idx}  (live)"

    def _on_play_toggled(self, paused: bool) -> None:
        self.btn_play.setText("Play" if paused else "Pause")
        self.play_toggle.emit(paused)

    # Public API used by MainWindow to keep UI in sync with the decoder.
    def update_position(self, idx: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(idx))
        self.slider.blockSignals(False)
        self.lbl_pos.setText(self._fmt(int(idx)))

    def is_paused(self) -> bool:
        return self.btn_play.isChecked()
