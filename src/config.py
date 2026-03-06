"""
Configuration module for the VSHORAD system.

Contains tier-specific model configurations, class taxonomies,
Swin-to-YOLO category mappings, ByteTrack parameters, and
feedback loop settings used across all system tiers.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# YOLO Detection Classes (12 meta-categories)
# =============================================================================

YOLO_CLASSES: Dict[int, str] = {
    0: "Fighter",
    1: "Helicopter",
    2: "Transport",
    3: "Bomber",
    4: "Special",
    5: "UAV_Fixed",
    6: "UAV_Rotor",
    7: "Missile",
    8: "Civilian",
    9: "Birds",
    10: "Decoys",
    11: "Unidentified",
}

# =============================================================================
# Swin Transformer Classes (56 fine-grained types)
# =============================================================================

SWIN_CLASSES: List[str] = [
    "AH-1", "AH-64", "AN-124", "AN-26", "A-10", "AV-8B",
    "B1", "B2", "B52",
    "C130", "C17", "C5", "CH-47",
    "E2", "E3", "E7", "EUROFIGHTER",
    "F-117", "F-14", "F-15", "F-16", "F-22", "F-35", "F-4", "FA-18",
    "Flanker_family", "Fulcrum_family", "GRIPEN",
    "H6", "IL-76",
    "J-10", "J-20", "J-31", "JF-17",
    "KA-52", "KC-135",
    "MI-24", "MI-28", "MI-8", "MIG-31", "MIRAGE-2000",
    "RAFALE",
    "SR-71", "SU-25", "SU-34", "SU-57",
    "TIGER", "TU-160", "TU-95",
    "U2", "UH60M",
    "V22", "VULCAN",
    "Y20", "YF-23", "Z10",
]

# =============================================================================
# Swin → YOLO Category Mapping
# =============================================================================

SWIN_TO_YOLO_CATEGORY: Dict[str, int] = {
    # FIGHTER (YOLO class 0)
    "F-16": 0, "F-15": 0, "F-22": 0, "F-35": 0, "F-14": 0, "F-4": 0,
    "FA-18": 0, "EUROFIGHTER": 0, "GRIPEN": 0, "RAFALE": 0,
    "MIRAGE-2000": 0, "J-10": 0, "J-20": 0, "J-31": 0, "JF-17": 0,
    "MIG-31": 0, "Fulcrum_family": 0, "Flanker_family": 0, "SU-57": 0,
    "YF-23": 0, "F-117": 0,
    # HELICOPTER (YOLO class 1)
    "AH-64": 1, "AH-1": 1, "MI-24": 1, "MI-28": 1, "KA-52": 1,
    "MI-8": 1, "CH-47": 1, "UH60M": 1, "TIGER": 1, "Z10": 1,
    # TRANSPORT (YOLO class 2)
    "C130": 2, "C17": 2, "C5": 2, "AN-124": 2, "AN-26": 2,
    "IL-76": 2, "Y20": 2,
    # BOMBER (YOLO class 3)
    "B1": 3, "B2": 3, "B52": 3, "TU-160": 3, "TU-95": 3,
    "VULCAN": 3, "H6": 3, "SU-34": 3,
    # SPECIAL (YOLO class 4)
    "E2": 4, "E3": 4, "E7": 4, "KC-135": 4, "U2": 4, "SR-71": 4,
    "V22": 4, "A-10": 4, "AV-8B": 4, "SU-25": 4,
}

# Classes that Swin can refine (fighter, helicopter, transport, bomber, special)
SWIN_SUPPORTED_YOLO_CLASSES: Set[int] = {0, 1, 2, 3, 4}

# Classes Swin should never attempt to classify
SWIN_UNSUPPORTED_YOLO_CLASSES: Set[int] = {5, 6, 7, 8, 9, 10, 11}


# =============================================================================
# Tier Enumeration
# =============================================================================

class Tier(str, Enum):
    """Available system deployment tiers."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EMBEDDED = "embedded"


# =============================================================================
# Tier-Specific Model Configuration
# =============================================================================

@dataclass
class YOLOConfig:
    """YOLO detection model configuration."""
    model: str
    imgsz: int
    epochs: int = 120
    patience: int = 25
    batch: int = 32
    optimizer: str = "AdamW"
    lr0: float = 0.002
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    # Augmentation
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.1
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    erasing: float = 0.2
    seed: int = 42


@dataclass
class SwinConfig:
    """Swin Transformer classification model configuration."""
    model_name: str
    img_size: int
    num_classes: int = 56
    pretrained: bool = True
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    drop_path_rate: float = 0.2
    label_smoothing: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.9999
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    seed: int = 42


@dataclass
class ByteTrackConfig:
    """ByteTrack multi-object tracker configuration."""
    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.55
    track_buffer: int = 90
    match_thresh: float = 0.92
    fuse_score: bool = True


@dataclass
class FeedbackConfig:
    """
    Feedback loop configuration controlling how YOLO detections
    and Swin classifications are fused together.
    """
    # Classification routing
    always_classify: List[int] = field(default_factory=lambda: [0, 3])
    classify_if_uncertain: List[int] = field(default_factory=lambda: [1, 2, 4])
    never_classify: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10, 11])
    uncertainty_threshold: float = 0.70

    # Swin EMA (Exponential Moving Average)
    swin_ema_alpha: float = 0.4
    swin_ema_min_samples: int = 2

    # Reclassification interval (frames)
    reclassify_interval: int = 25

    # Bounding box smoothing
    bbox_smoothing: bool = True
    bbox_alpha: float = 0.6

    # Re-Identification
    reid_enabled: bool = True
    reid_max_frames: int = 45
    reid_max_distance: float = 200.0
    reid_velocity_mult: float = 1.5
    reid_same_class_required: bool = True

    # Track persistence
    track_hold_frames: int = 60
    min_bbox_size: int = 20

    # Fusion thresholds
    swin_override_threshold: float = 0.80
    yolo_weak_threshold: float = 0.50

    # YOLO stability
    yolo_stable_frames: int = 8

    # Unidentified detection
    unidentified_window: int = 30
    unidentified_max_changes: int = 10


# =============================================================================
# Pre-built Tier Configurations
# =============================================================================

TIER_CONFIGS = {
    Tier.STRATEGIC: {
        "yolo": YOLOConfig(model="yolov8l.pt", imgsz=1280, batch=32, lr0=0.002, mixup=0.15, copy_paste=0.1),
        "swin": SwinConfig(model_name="swin_base_patch4_window12_384", img_size=384, batch_size=64, lr=2e-4),
        "bytetrack": ByteTrackConfig(),
        "feedback": FeedbackConfig(),
    },
    Tier.TACTICAL: {
        "yolo": YOLOConfig(model="yolov8m.pt", imgsz=960, batch=16, lr0=0.001, mixup=0.1, copy_paste=0.0),
        "swin": SwinConfig(model_name="swin_small_patch4_window7_224", img_size=224, batch_size=32, lr=1e-4, num_workers=4),
        "bytetrack": ByteTrackConfig(),
        "feedback": FeedbackConfig(),
    },
    Tier.EMBEDDED: {
        "yolo": YOLOConfig(model="yolov8m.pt", imgsz=640, batch=8),
        "swin": SwinConfig(model_name="swin_small_patch4_window7_224", img_size=224, batch_size=16),
        "bytetrack": ByteTrackConfig(),
        "feedback": FeedbackConfig(),
    },
}


def get_tier_config(tier: Tier) -> dict:
    """Retrieve the full configuration dictionary for a given tier."""
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Unknown tier: {tier}. Choose from {list(Tier)}")
    return TIER_CONFIGS[tier]
