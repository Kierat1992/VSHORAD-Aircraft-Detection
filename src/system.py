"""
VSHORAD System — main orchestrator.

Combines YOLOv8 detection, ByteTrack tracking, Swin Transformer
classification, and sensor fusion into a unified frame-by-frame
processing pipeline for aircraft detection and identification.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.classification import SwinClassifier
from src.config import (
    SWIN_CLASSES,
    SWIN_TO_YOLO_CATEGORY,
    YOLO_CLASSES,
    ByteTrackConfig,
    FeedbackConfig,
    Tier,
    get_tier_config,
)
from src.detection import YOLODetector
from src.fusion import FusionEngine
from src.tracking import TrackManager


class VSHORADSystem:
    """
    Multi-tier aircraft detection, tracking, and classification system.

    Architecture:
        Frame → YOLOv8 (detection + ByteTrack) → Swin (classification) → Fusion → Output

    The system processes video frames sequentially, maintaining persistent
    track state across frames. Each detection is assigned a stable track ID
    via ByteTrack, optionally classified by Swin Transformer for fine-grained
    identification, and fused using confidence-based rules.

    Features:
        - Multi-object detection with 12 meta-categories
        - Fine-grained classification into 56 aircraft types
        - ByteTrack-based tracking with Re-Identification
        - Swin probability EMA for temporal smoothing
        - Unidentified target detection via label oscillation analysis
        - Bounding box temporal smoothing

    Args:
        yolo_weights: Path to YOLOv8 weights.
        swin_weights: Path to Swin Transformer checkpoint.
        tier: System tier (determines model architecture and parameters).
        feedback_config: Override default feedback loop parameters.
        bytetrack_config: Override default ByteTrack parameters.
        device: Inference device ('cuda' or 'cpu').

    Example:
        >>> system = VSHORADSystem(
        ...     yolo_weights="weights/yolov8l_strategic.pt",
        ...     swin_weights="weights/swin_base_strategic.pth",
        ...     tier=Tier.STRATEGIC,
        ... )
        >>> results = system.process_frame(frame, frame_idx=0)
        >>> for det in results["detections"]:
        ...     print(f"Track #{det['track_id']}: {det['final_label']} ({det['final_conf']:.2f})")
    """

    def __init__(
        self,
        yolo_weights: str,
        swin_weights: str,
        tier: Tier = Tier.STRATEGIC,
        feedback_config: Optional[FeedbackConfig] = None,
        bytetrack_config: Optional[ByteTrackConfig] = None,
        device: str = "cuda",
    ):
        tier_cfg = get_tier_config(tier)
        yolo_cfg = tier_cfg["yolo"]
        swin_cfg = tier_cfg["swin"]
        bt_cfg = bytetrack_config or tier_cfg["bytetrack"]
        fb_cfg = feedback_config or tier_cfg["feedback"]

        self.tier = tier
        self.device = device

        # Initialize components
        print(f"[VSHORAD] Initializing {tier.value} tier on {device}")

        print(f"  Loading YOLO ({yolo_cfg.model}, {yolo_cfg.imgsz}px)...")
        self.detector = YOLODetector(
            weights_path=yolo_weights,
            conf_threshold=yolo_cfg.conf_threshold,
            iou_threshold=yolo_cfg.iou_threshold,
            bytetrack_config=bt_cfg,
            device=device,
        )

        print(f"  Loading Swin ({swin_cfg.model_name}, {swin_cfg.img_size}px)...")
        self.classifier = SwinClassifier(
            weights_path=swin_weights,
            model_name=swin_cfg.model_name,
            img_size=swin_cfg.img_size,
            num_classes=swin_cfg.num_classes,
            class_names=SWIN_CLASSES,
            device=device,
        )

        self.tracker = TrackManager(fb_cfg)
        self.fusion = FusionEngine(fb_cfg)
        self.feedback_config = fb_cfg

        print(f"  System ready! Swin classes: {swin_cfg.num_classes}")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Process a single video frame through the full pipeline.

        Pipeline steps:
            1. YOLO detection + ByteTrack tracking
            2. Track state update (velocity, bbox smoothing, stability)
            3. Conditional Swin classification with EMA
            4. YOLO/Swin fusion
            5. Unidentified target detection
            6. Lost track management and Re-ID bookkeeping

        Args:
            frame: BGR image as numpy array (H, W, 3).
            frame_idx: Frame index for temporal bookkeeping.

        Returns:
            Dictionary containing:
                - 'frame_idx': Current frame index
                - 'detections': List of detection dictionaries
                - 'timing': Dict with 'yolo_ms', 'swin_ms', 'total_ms'
        """
        results: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "detections": [],
            "timing": {},
        }

        # --- Step 1: Detection + Tracking ---
        detections, yolo_ms = self.detector.detect(frame)
        results["timing"]["yolo_ms"] = yolo_ms

        swin_time = 0.0
        active_track_ids = set()

        for det in detections:
            active_track_ids.add(det.track_id)

            # --- Step 2: Track state management ---
            track = self.tracker.get_or_create_track(
                det.track_id, det.bbox, det.yolo_class, frame_idx
            )
            track.last_seen_frame = frame_idx
            track.total_detections += 1

            self.tracker.update_velocity(track, det.bbox)
            smooth_box = self.tracker.smooth_bbox(track, det.bbox)
            self.tracker.check_yolo_stable(track, det.yolo_class)
            track.yolo_class = det.yolo_class

            sx1, sy1, sx2, sy2 = smooth_box
            bbox_w = sx2 - sx1
            bbox_h = sy2 - sy1

            # Build output detection dict
            output = {
                "track_id": det.track_id,
                "bbox": list(smooth_box),
                "bbox_raw": list(det.bbox),
                "yolo_class": det.yolo_class,
                "yolo_label": det.yolo_label,
                "yolo_conf": det.yolo_conf,
                "final_label": det.yolo_label,
                "final_conf": det.yolo_conf,
                "source": "YOLO",
                "swin_used": False,
                "swin_label": None,
                "swin_conf": 0.0,
                "swin_detail": None,
                "yolo_stable": track.yolo_stable,
                "swin_samples": track.swin_sample_count,
                "is_unidentified": False,
            }

            # --- Step 3: Conditional Swin classification ---
            min_size = self.feedback_config.min_bbox_size
            can_classify = (
                self.fusion.should_classify(det.yolo_class, det.yolo_conf, track)
                and bbox_w >= min_size
                and bbox_h >= min_size
            )

            if can_classify:
                if self.fusion.needs_reclassification(track):
                    # Run Swin inference on the crop
                    x1, y1, x2, y2 = det.bbox
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        cls_result, cls_ms = self.classifier.classify_timed(crop_rgb)
                        swin_time += cls_ms

                        # Update EMA
                        ema_cls, ema_conf, ema_label = self.fusion.update_swin_ema(
                            track, cls_result.probabilities, SWIN_CLASSES
                        )
                        track.frame_count = 0

                        # --- Step 4: Fusion ---
                        fusion_result = self.fusion.fuse(
                            det.yolo_class, det.yolo_conf,
                            ema_cls, ema_conf, ema_label, track,
                        )
                        track.final_label = fusion_result.final_label
                        track.final_conf = fusion_result.final_conf

                        output.update({
                            "swin_class": cls_result.class_index,
                            "swin_label": cls_result.class_name,
                            "swin_conf": cls_result.confidence,
                            "swin_ema_label": ema_label,
                            "swin_ema_conf": ema_conf,
                            "swin_top5": cls_result.top5,
                            "final_label": fusion_result.final_label,
                            "final_conf": fusion_result.final_conf,
                            "source": fusion_result.source,
                            "swin_used": True,
                            "swin_detail": fusion_result.swin_detail or ema_label,
                            "swin_samples": track.swin_sample_count,
                        })
                else:
                    # Use cached Swin result
                    if track.last_swin_label is not None:
                        ema_probs = track.swin_probs_ema
                        ema_cls = int(np.argmax(ema_probs)) if ema_probs is not None else 0

                        fusion_result = self.fusion.fuse(
                            det.yolo_class, det.yolo_conf,
                            ema_cls, track.last_swin_conf,
                            track.last_swin_label, track,
                        )

                        output.update({
                            "swin_label": track.last_swin_label,
                            "swin_conf": track.last_swin_conf,
                            "swin_ema_label": track.last_swin_label,
                            "swin_ema_conf": track.last_swin_conf,
                            "final_label": fusion_result.final_label,
                            "final_conf": fusion_result.final_conf,
                            "source": fusion_result.source + " (cached)",
                            "swin_used": True,
                            "swin_detail": fusion_result.swin_detail or track.last_swin_label,
                            "swin_samples": track.swin_sample_count,
                        })

                    track.frame_count = getattr(track, "frame_count", 0) + 1

            # Propagate cached Swin info even if not classified this frame
            if track.last_swin_label and not output.get("swin_label"):
                output["swin_label"] = track.last_swin_label
                output["swin_conf"] = track.last_swin_conf
                output["swin_detail"] = track.last_swin_label

            # --- Step 5: Unidentified detection ---
            is_unidentified = self.tracker.check_unidentified(
                track, output["final_label"]
            )
            if is_unidentified:
                output["final_label"] = "Unidentified"
                output["source"] = "UNSTABLE"
                output["is_unidentified"] = True

            results["detections"].append(output)

        # --- Step 6: Track lifecycle management ---
        self.tracker.update_lost_tracks(active_track_ids, frame_idx)

        results["timing"]["swin_ms"] = swin_time
        results["timing"]["total_ms"] = yolo_ms + swin_time
        return results

    def reset(self) -> None:
        """Reset all tracking state. Call between videos."""
        self.tracker.reset()
        self.detector.reset()

    @property
    def stats(self) -> dict:
        """Return current system statistics."""
        return {
            "tier": self.tier.value,
            "tracking": self.tracker.stats,
        }
