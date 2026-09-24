"""Qt window that shows a live video stream, always keeping the whole frame visible."""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .camera import Camera
from .grabber import FrameGrabber, FrameProcessor

QUIT_KEYS = {Qt.Key.Key_Q, Qt.Key.Key_Escape}
INITIAL_SCREEN_FRACTION = 0.7


def fit_rect(frame_width: int, frame_height: int, area_width: int, area_height: int) -> QRect:
    """Largest rect with the frame's aspect ratio that fits in the area, centered."""
    scale = min(area_width / frame_width, area_height / frame_height)
    width = round(frame_width * scale)
    height = round(frame_height * scale)
    return QRect((area_width - width) // 2, (area_height - height) // 2, width, height)


def lock_aspect_ratio(widget: QWidget, width: int, height: int) -> bool:
    """Make macOS keep the window content at width:height while the user resizes it."""
    if sys.platform != "darwin" or QApplication.platformName() != "cocoa":
        return False
    try:
        import objc
    except ImportError:
        return False

    view = objc.objc_object(c_void_p=int(widget.winId()))
    window = view.window()
    if window is None:
        return False
    window.setContentAspectRatio_((width, height))
    return True


class VideoWindow(QWidget):
    """Displays frames pushed via ``frame_ready`` (safe to emit from any thread)."""

    frame_ready = Signal(object)
    stream_ended = Signal()
    closed = Signal()

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._frame: np.ndarray | None = None
        self._image: QImage | None = None
        self._frame_size: tuple[int, int] | None = None
        self.frame_ready.connect(self.set_frame)
        self.stream_ended.connect(self.close)

    def set_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        # QImage does not copy the pixels, so the array must stay referenced.
        self._frame = frame
        self._image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
        if self._frame_size != (width, height):
            self._frame_size = (width, height)
            self._fit_window_to_frame(width, height)
        self.update()

    def _fit_window_to_frame(self, width: int, height: int) -> None:
        screen = self.screen().availableGeometry()
        target = fit_rect(
            width,
            height,
            int(screen.width() * INITIAL_SCREEN_FRACTION),
            int(screen.height() * INITIAL_SCREEN_FRACTION),
        )
        self.resize(target.width(), target.height())
        lock_aspect_ratio(self, width, height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            target = fit_rect(self._image.width(), self._image.height(), self.width(), self.height())
            painter.drawImage(target, self._image)
        painter.end()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in QUIT_KEYS:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


def run_video_app(camera: Camera, title: str, process: FrameProcessor | None = None) -> None:
    """Open the camera, show its stream in a VideoWindow and block until the window closes.

    Frames are read (and passed through ``process``) in a background thread.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    with camera:
        window = VideoWindow(title)
        grabber = FrameGrabber(
            camera,
            on_frame=window.frame_ready.emit,
            process=process,
            on_stop=lambda error: window.stream_ended.emit(),
        )
        window.closed.connect(grabber.stop)
        window.show()
        grabber.start()
        app.exec()
        grabber.stop()
        grabber.join(timeout=2)
    if grabber.error is not None:
        raise grabber.error
