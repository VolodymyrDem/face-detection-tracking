"""Background camera reading, so the GUI thread never waits for the camera."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from .camera import Camera

FrameCallback = Callable[[np.ndarray], None]
FrameProcessor = Callable[[np.ndarray], np.ndarray]


class FrameGrabber:
    """Reads frames from an opened Camera in a daemon thread and hands them to a callback.

    ``process`` (e.g. detection + drawing) also runs in that thread, keeping heavy work
    off the GUI thread. ``on_stop`` is called once the thread ends, with the error that
    stopped it or None.
    """

    def __init__(
        self,
        camera: Camera,
        on_frame: FrameCallback,
        process: FrameProcessor | None = None,
        on_stop: Callable[[BaseException | None], None] | None = None,
    ) -> None:
        self._camera = camera
        self._on_frame = on_frame
        self._process = process
        self._on_stop = on_stop
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FrameGrabber", daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            for frame in self._camera.frames():
                if self._stop_event.is_set():
                    break
                if self._process is not None:
                    frame = self._process(frame)
                self._on_frame(frame)
        except BaseException as error:  # noqa: BLE001 - reported to the caller via on_stop
            self.error = error
        finally:
            if self._on_stop is not None:
                self._on_stop(self.error)
