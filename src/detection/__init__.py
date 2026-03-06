"""
Detection module wrapping Ultralytics YOLOv8 for aircraft detection.

Provides a clean interface for loading YOLO models, running inference
with integrated ByteTrack tracking, and extracting structured results.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.config import ByteTrackConfig, YOLO_CLASSES


@dataclass
class Detection:
    """Single detection result from YOLO + ByteTrack."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    yolo_class: int
    yolo_conf: float

    @property
    def yolo_label(self) -> str:
        return YOLO_CLASSES.get(self.yolo_class, "Unknown")

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class YOLODetector:
    """
    YOLOv8 aircraft detector with integrated ByteTrack tracking.

    Args:
        weights_path: Path to YOLOv8 weights (.pt or .engine).
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        bytetrack_config: ByteTrack tracker configuration.
        device: Inference device ('cuda', 'cpu', or device index).
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        bytetrack_config: Optional[ByteTrackConfig] = None,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Write ByteTrack config to a temporary YAML file
        self._tracker_config_path = self._write_tracker_config(
            bytetrack_config or ByteTrackConfig()
        )

    def _write_tracker_config(self, config: ByteTrackConfig) -> str:
        """Serialize ByteTrack config to a YAML file for Ultralytics."""
        config_dict = {
            "tracker_type": "bytetrack",
            "track_high_thresh": config.track_high_thresh,
            "track_low_thresh": config.track_low_thresh,
            "new_track_thresh": config.new_track_thresh,
            "track_buffer": config.track_buffer,
            "match_thresh": config.match_thresh,
            "fuse_score": config.fuse_score,
        }
        import tempfile
        config_path = Path(tempfile.gettempdir()) / "vshorad_bytetrack.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        return str(config_path)

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            Tuple of (list of Detection objects, inference time in ms).
        """
        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_config_path,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, track_ids, classes, confs):
                detections.append(Detection(
                    track_id=int(tid),
                    bbox=tuple(box.tolist()),
                    yolo_class=int(cls),
                    yolo_conf=float(conf),
                ))

        return detections, elapsed_ms

    def reset(self) -> None:
        """Reset the tracker state (call between videos)."""
        self.model.predictor = None
