from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CameraFrame:
    width: int
    height: int
    timestamp_seconds: float
    source_name: str = "webcam"


class CameraSource:
    """Scaffold interface for future OpenCV camera integration."""

    def read_frame(self) -> CameraFrame:
        raise NotImplementedError("Camera integration is not implemented yet.")


class OpenCVCameraSource(CameraSource):
    """Thin wrapper around cv2.VideoCapture for demo use."""

    def __init__(self, capture: Any, source_name: str = "webcam") -> None:
        self.capture = capture
        self.source_name = source_name

    def read(self) -> tuple[bool, Any]:
        return self.capture.read()
