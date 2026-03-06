"""
Sensor fusion module for combining YOLO detection and Swin classification.

Implements the decision logic for:
    - When to invoke Swin classification (routing rules)
    - Swin probability EMA (Exponential Moving Average) updates
    - YOLO/Swin fusion rules with confidence-based overrides
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config import (
    FeedbackConfig,
    SWIN_SUPPORTED_YOLO_CLASSES,
    SWIN_TO_YOLO_CATEGORY,
    YOLO_CLASSES,
)
from src.tracking import TrackState


@dataclass
class FusionResult:
    """Output of the sensor fusion decision."""
    final_label: str
    final_conf: float
    source: str           # 'YOLO', 'SWIN [F-16]', 'SWIN [F-16] (cached)', 'UNSTABLE'
    swin_detail: Optional[str] = None


class FusionEngine:
    """
    Manages the fusion of YOLO detection and Swin classification results.

    The fusion engine decides:
        1. Whether a detection should be classified by Swin.
        2. How to merge Swin EMA probabilities with YOLO predictions.
        3. Whether Swin should override YOLO based on confidence thresholds.

    Args:
        config: Feedback loop configuration.
    """

    def __init__(self, config: FeedbackConfig):
        self.config = config

    # -------------------------------------------------------------------------
    # Classification routing
    # -------------------------------------------------------------------------

    def should_classify(
        self,
        yolo_class: int,
        yolo_conf: float,
        track: TrackState,
    ) -> bool:
        """
        Determine whether a detection should be sent to Swin for
        fine-grained classification.

        Rules:
            - Never classify classes Swin doesn't cover (UAV, Missile, etc.)
            - Always classify Fighters and Bombers (high-priority targets)
            - Classify uncertain Helicopters, Transport, Special only
              when YOLO confidence is below the uncertainty threshold
        """
        cfg = self.config

        if yolo_class in cfg.never_classify:
            return False

        if yolo_class in cfg.always_classify:
            return track.yolo_stable or track.swin_sample_count > 0

        if yolo_class in cfg.classify_if_uncertain:
            return (
                yolo_conf < cfg.uncertainty_threshold
                and track.yolo_stable
            )

        return False

    def needs_reclassification(self, track: TrackState) -> bool:
        """Check if a track is due for periodic Swin reclassification."""
        return (
            track.frame_count >= self.config.reclassify_interval
            or track.swin_sample_count == 0
        )

    # -------------------------------------------------------------------------
    # Swin EMA updates
    # -------------------------------------------------------------------------

    def update_swin_ema(
        self,
        track: TrackState,
        new_probs: np.ndarray,
        class_names: list,
    ) -> Tuple[int, float, str]:
        """
        Update the Swin probability EMA for a track.

        Uses exponential moving average to smooth predictions over
        multiple frames, reducing single-frame classification noise.

        Returns:
            Tuple of (predicted class index, EMA confidence, class name).
        """
        alpha = self.config.swin_ema_alpha

        if track.swin_probs_ema is None:
            track.swin_probs_ema = new_probs.copy()
        else:
            track.swin_probs_ema = (
                alpha * new_probs + (1 - alpha) * track.swin_probs_ema
            )
        track.swin_sample_count += 1

        ema_pred = int(np.argmax(track.swin_probs_ema))
        ema_conf = float(track.swin_probs_ema[ema_pred])
        ema_label = class_names[ema_pred]

        track.last_swin_label = ema_label
        track.last_swin_conf = ema_conf

        return ema_pred, ema_conf, ema_label

    # -------------------------------------------------------------------------
    # Fusion rules
    # -------------------------------------------------------------------------

    def fuse(
        self,
        yolo_class: int,
        yolo_conf: float,
        swin_class: int,
        swin_conf: float,
        swin_label: str,
        track: TrackState,
    ) -> FusionResult:
        """
        Apply fusion rules to combine YOLO and Swin predictions.

        Decision logic:
            1. If YOLO class is not supported by Swin → use YOLO.
            2. If Swin confidence exceeds override threshold AND YOLO
               confidence is below weak threshold → Swin overrides.
            3. If Swin and YOLO agree on category → Swin provides
               fine-grained label with Swin confidence.
            4. If Swin strongly disagrees (>0.90 confidence) → Swin
               overrides YOLO category entirely.
            5. Otherwise → use YOLO label.

        Returns:
            FusionResult with the final label, confidence, and source.
        """
        cfg = self.config
        yolo_label = YOLO_CLASSES[yolo_class]

        # Swin cannot classify this category
        if yolo_class not in SWIN_SUPPORTED_YOLO_CLASSES:
            return FusionResult(
                final_label=yolo_label,
                final_conf=yolo_conf,
                source="YOLO",
            )

        # Map Swin fine-grained class to YOLO meta-category
        swin_yolo_cat = SWIN_TO_YOLO_CATEGORY.get(swin_label)

        # Check if Swin can override
        can_override = (
            swin_conf > cfg.swin_override_threshold
            and yolo_conf < cfg.yolo_weak_threshold
        )

        if can_override and swin_yolo_cat is not None:
            if swin_yolo_cat == yolo_class:
                # Swin agrees with YOLO category → refine with detail
                return FusionResult(
                    final_label=yolo_label,
                    final_conf=swin_conf,
                    source=f"SWIN [{swin_label}]",
                    swin_detail=swin_label,
                )
            elif swin_conf > 0.90:
                # Swin strongly disagrees → override YOLO category
                overridden_label = YOLO_CLASSES[swin_yolo_cat]
                return FusionResult(
                    final_label=overridden_label,
                    final_conf=swin_conf,
                    source=f"SWIN [{swin_label}]",
                    swin_detail=swin_label,
                )

        # Default: keep YOLO prediction
        return FusionResult(
            final_label=yolo_label,
            final_conf=yolo_conf,
            source="YOLO",
            swin_detail=swin_label,
        )
