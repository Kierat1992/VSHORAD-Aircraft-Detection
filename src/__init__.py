"""
VSHORAD Aircraft Detection & Classification System
====================================================

A multi-tier computer vision pipeline for Very Short Range Air Defense,
combining YOLOv8 detection, ByteTrack tracking, and Swin Transformer
classification with intelligent sensor fusion.

Tiers:
    - Strategic: YOLOv8l @ 1280px + Swin-Base @ 384px (A100 GPU)
    - Tactical:  YOLOv8m @ 960px  + Swin-Small @ 224px (L4/T4 GPU)
    - Embedded:  TensorRT FP16/INT8 exports for edge deployment

Author: Jędrzej Rychter
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Jędrzej Rychter"

from src.system import VSHORADSystem

__all__ = ["VSHORADSystem"]
