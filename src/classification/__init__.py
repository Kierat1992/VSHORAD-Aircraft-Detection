"""
Classification module wrapping Swin Transformer for fine-grained
aircraft type identification.

Supports loading weights from both Strategic and Tactical checkpoint
formats, and provides a unified inference interface with top-K results.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from albumentations.pytorch import ToTensorV2

from src.config import SWIN_CLASSES


@dataclass
class ClassificationResult:
    """Single classification output from Swin Transformer."""
    class_index: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    top5: List[Tuple[str, float]]


class SwinClassifier:
    """
    Swin Transformer classifier for fine-grained aircraft identification.

    Loads a pre-trained Swin model from a checkpoint file and provides
    inference on cropped aircraft images. Handles both Strategic
    (model_state_dict / ema_state_dict) and Tactical (model / ema)
    checkpoint key formats.

    Args:
        weights_path: Path to the Swin checkpoint (.pth file).
        model_name: Timm model identifier.
        img_size: Input image resolution.
        num_classes: Number of output classes.
        class_names: Ordered list of class names.
        device: Inference device.
        prefer_ema: If True, load EMA weights when available.
    """

    def __init__(
        self,
        weights_path: str,
        model_name: str = "swin_base_patch4_window12_384",
        img_size: int = 384,
        num_classes: int = 56,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
        prefer_ema: bool = True,
    ):
        self.device = device
        self.img_size = img_size
        self.class_names = class_names or SWIN_CLASSES
        self.num_classes = num_classes

        # Build model architecture
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        ).to(device)

        # Load weights (handle both checkpoint formats)
        self._load_weights(weights_path, prefer_ema)
        self.model.eval()

        # Preprocessing transform (deterministic, no augmentation)
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _load_weights(self, weights_path: str, prefer_ema: bool) -> None:
        """
        Load model weights from checkpoint, handling multiple formats.

        Strategic checkpoints use keys: 'model_state_dict', 'ema_state_dict'
        Tactical checkpoints use keys: 'model', 'ema'
        """
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Determine the correct state dict key
        state_dict = None
        if prefer_ema:
            # Try EMA keys first (preferred for inference)
            for key in ("ema_state_dict", "ema"):
                if key in checkpoint and checkpoint[key] is not None:
                    state_dict = checkpoint[key]
                    break

        if state_dict is None:
            # Fall back to model keys
            for key in ("model_state_dict", "model"):
                if key in checkpoint and checkpoint[key] is not None:
                    state_dict = checkpoint[key]
                    break

        if state_dict is None:
            # Assume the checkpoint IS the state dict
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classify a single cropped aircraft image.

        Args:
            crop: RGB image as numpy array (H, W, 3).

        Returns:
            ClassificationResult with predicted class, confidence,
            full probability distribution, and top-5 predictions.
        """
        augmented = self.transform(image=crop)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        output = self.model(tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

        # Top-5 predictions
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5 = [
            (self.class_names[idx], float(probs[idx]))
            for idx in top5_indices
        ]

        return ClassificationResult(
            class_index=pred_idx,
            class_name=self.class_names[pred_idx],
            confidence=pred_conf,
            probabilities=probs,
            top5=top5,
        )

    @torch.no_grad()
    def classify_timed(self, crop: np.ndarray) -> Tuple[ClassificationResult, float]:
        """
        Classify with timing information.

        Returns:
            Tuple of (ClassificationResult, elapsed time in ms).
        """
        t0 = time.perf_counter()
        result = self.classify(crop)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms
