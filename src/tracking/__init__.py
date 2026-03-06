"""
Track management module for multi-object tracking state.

Handles track lifecycle (creation, update, loss, re-identification),
velocity estimation, bounding box smoothing, YOLO class stability
checks, and unidentified target detection via label change analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import FeedbackConfig


@dataclass
class TrackState:
    """Internal state maintained for each tracked object."""
    # Spatial
    last_center: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    smooth_bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # YOLO stability
    yolo_class_history: List[int] = field(default_factory=list)
    yolo_stable: bool = False
    yolo_class: Optional[int] = None

    # Swin classification (EMA)
    swin_probs_ema: Optional[np.ndarray] = None
    swin_sample_count: int = 0
    last_swin_label: Optional[str] = None
    last_swin_conf: float = 0.0

    # Final fused result
    final_label: Optional[str] = None
    final_conf: float = 0.0

    # Lifecycle
    frame_count: int = 0
    last_seen_frame: int = 0
    created_frame: int = 0
    total_detections: int = 0

    # Unidentified detection
    class_change_history: List[str] = field(default_factory=list)
    is_unidentified: bool = False


class TrackManager:
    """
    Manages the lifecycle and state of all tracked objects.

    Responsibilities:
        - Track creation and initialization
        - Velocity estimation with exponential smoothing
        - Bounding box temporal smoothing
        - YOLO class stability assessment
        - Unidentified target detection (excessive label oscillation)
        - Re-identification of lost tracks using velocity prediction
        - Automatic cleanup of stale tracks

    Args:
        config: Feedback loop configuration parameters.
    """

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, dict] = {}
        self.reid_count: int = 0

    # -------------------------------------------------------------------------
    # Track creation
    # -------------------------------------------------------------------------

    def get_or_create_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        yolo_class: int,
        frame_idx: int,
    ) -> TrackState:
        """
        Retrieve existing track or create a new one. Attempts Re-ID
        against lost tracks before creating a fresh state.
        """
        if track_id not in self.tracks:
            restored = self._try_reid(bbox, yolo_class, frame_idx)
            if restored is not None:
                self.tracks[track_id] = restored
            else:
                self.tracks[track_id] = self._init_track(bbox, frame_idx)

        return self.tracks[track_id]

    def _init_track(
        self,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> TrackState:
        """Initialize a new track state from a detection."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return TrackState(
            last_center=(cx, cy),
            smooth_bbox=[float(x1), float(y1), float(x2), float(y2)],
            last_seen_frame=frame_idx,
            created_frame=frame_idx,
        )

    # -------------------------------------------------------------------------
    # Spatial updates
    # -------------------------------------------------------------------------

    def update_velocity(
        self,
        track: TrackState,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """Update track velocity using exponential smoothing."""
        x1, y1, x2, y2 = bbox
        new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2
        old_cx, old_cy = track.last_center

        vx = new_cx - old_cx
        vy = new_cy - old_cy
        alpha = 0.5

        old_vx, old_vy = track.velocity
        track.velocity = (
            alpha * vx + (1 - alpha) * old_vx,
            alpha * vy + (1 - alpha) * old_vy,
        )
        track.last_center = (new_cx, new_cy)

    def smooth_bbox(
        self,
        track: TrackState,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to bounding box coordinates."""
        if not self.config.bbox_smoothing:
            return bbox

        alpha = self.config.bbox_alpha
        new = [float(x) for x in bbox]
        old = track.smooth_bbox
        smoothed = [alpha * new[i] + (1 - alpha) * old[i] for i in range(4)]
        track.smooth_bbox = smoothed
        return tuple(int(round(x)) for x in smoothed)

    # -------------------------------------------------------------------------
    # YOLO class stability
    # -------------------------------------------------------------------------

    def check_yolo_stable(self, track: TrackState, yolo_class: int) -> bool:
        """
        Assess whether the YOLO class assignment has stabilized.

        A track is considered stable if the same class has been
        predicted for `yolo_stable_frames` consecutive frames.
        """
        history = track.yolo_class_history
        history.append(yolo_class)

        max_len = self.config.yolo_stable_frames
        if len(history) > max_len:
            history.pop(0)

        track.yolo_stable = (
            len(history) >= max_len and len(set(history)) == 1
        )
        return track.yolo_stable

    # -------------------------------------------------------------------------
    # Unidentified detection
    # -------------------------------------------------------------------------

    def check_unidentified(self, track: TrackState, label: str) -> bool:
        """
        Detect unstable tracks that oscillate between labels.

        If the number of label changes within a sliding window exceeds
        the threshold, the track is marked as unidentified. It can
        recover once the label stabilizes.
        """
        window = self.config.unidentified_window
        max_changes = self.config.unidentified_max_changes

        history = track.class_change_history
        history.append(label)
        if len(history) > window:
            history.pop(0)

        if len(history) >= window // 2:
            changes = sum(
                1 for i in range(1, len(history))
                if history[i] != history[i - 1]
            )
            if changes > max_changes:
                track.is_unidentified = True
            elif changes <= max_changes // 3 and len(history) >= window:
                track.is_unidentified = False

        return track.is_unidentified

    # -------------------------------------------------------------------------
    # Re-Identification
    # -------------------------------------------------------------------------

    def _try_reid(
        self,
        bbox: Tuple[int, int, int, int],
        yolo_class: int,
        frame_idx: int,
    ) -> Optional[TrackState]:
        """
        Attempt to re-identify a new detection as a previously lost track.

        Uses velocity-based position prediction and distance matching
        to associate new detections with lost tracks.
        """
        cfg = self.config
        if not cfg.reid_enabled:
            return None

        x1, y1, x2, y2 = bbox
        new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2

        best_match_id = None
        best_score = float("inf")

        for tid, info in list(self.lost_tracks.items()):
            frames_lost = frame_idx - info["last_seen_frame"]

            # Expire old lost tracks
            if frames_lost > cfg.reid_max_frames:
                del self.lost_tracks[tid]
                continue

            # Require same YOLO class
            if cfg.reid_same_class_required and info.get("yolo_class") != yolo_class:
                continue

            # Predict position using velocity
            old_cx, old_cy = info["last_center"]
            vx, vy = info["velocity"]
            pred_cx = old_cx + vx * frames_lost * cfg.reid_velocity_mult
            pred_cy = old_cy + vy * frames_lost * cfg.reid_velocity_mult

            dist = ((new_cx - pred_cx) ** 2 + (new_cy - pred_cy) ** 2) ** 0.5

            if dist < cfg.reid_max_distance and dist < best_score:
                best_score = dist
                best_match_id = tid

        if best_match_id is None:
            return None

        info = self.lost_tracks.pop(best_match_id)
        self.reid_count += 1

        return TrackState(
            last_center=(new_cx, new_cy),
            velocity=info.get("velocity", (0, 0)),
            smooth_bbox=[float(x1), float(y1), float(x2), float(y2)],
            swin_probs_ema=info.get("swin_probs_ema"),
            swin_sample_count=info.get("swin_sample_count", 0),
            last_swin_label=info.get("last_swin_label"),
            last_swin_conf=info.get("last_swin_conf", 0.0),
            last_seen_frame=frame_idx,
            created_frame=frame_idx,
            class_change_history=info.get("class_change_history", []),
            is_unidentified=info.get("is_unidentified", False),
        )

    # -------------------------------------------------------------------------
    # Lifecycle management
    # -------------------------------------------------------------------------

    def update_lost_tracks(self, active_ids: set, frame_idx: int) -> None:
        """
        Move disappeared tracks to the lost pool and clean up stale entries.
        """
        cfg = self.config

        for tid in list(self.tracks.keys()):
            if tid in active_ids:
                continue

            track = self.tracks[tid]
            frames_since = frame_idx - track.last_seen_frame

            # Move to lost tracks for potential Re-ID
            if frames_since <= cfg.reid_max_frames and tid not in self.lost_tracks:
                self.lost_tracks[tid] = {
                    "last_center": track.last_center,
                    "velocity": track.velocity,
                    "last_seen_frame": track.last_seen_frame,
                    "yolo_class": track.yolo_class,
                    "swin_probs_ema": track.swin_probs_ema,
                    "last_swin_label": track.last_swin_label,
                    "last_swin_conf": track.last_swin_conf,
                    "swin_sample_count": track.swin_sample_count,
                    "class_change_history": track.class_change_history,
                    "is_unidentified": track.is_unidentified,
                }

            # Remove fully expired tracks
            if frames_since > cfg.track_hold_frames:
                del self.tracks[tid]

        # Clean expired lost tracks
        for tid in list(self.lost_tracks.keys()):
            if frame_idx - self.lost_tracks[tid]["last_seen_frame"] > cfg.reid_max_frames:
                del self.lost_tracks[tid]

    def reset(self) -> None:
        """Clear all tracking state."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.reid_count = 0

    @property
    def stats(self) -> dict:
        """Return tracking statistics."""
        return {
            "active_tracks": len(self.tracks),
            "lost_tracks": len(self.lost_tracks),
            "total_reids": self.reid_count,
        }
