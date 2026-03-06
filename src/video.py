"""
Video processing module for running the VSHORAD system on video files.

Provides functions for frame-by-frame processing with visualization,
video output writing, and JSON result export.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.system import VSHORADSystem


# =============================================================================
# Color scheme for visualization
# =============================================================================

COLORS = {
    "swin": (0, 255, 255),       # Yellow — Swin-classified
    "yolo_stable": (0, 255, 0),  # Green — YOLO stable
    "yolo": (0, 200, 0),         # Dark green — YOLO unstable
    "unidentified": (0, 0, 255), # Red — Unidentified
}


def _get_detection_color(detection: dict) -> Tuple[int, int, int]:
    """Select visualization color based on detection source."""
    if detection.get("is_unidentified"):
        return COLORS["unidentified"]
    if "SWIN" in detection.get("source", ""):
        return COLORS["swin"]
    if detection.get("yolo_stable"):
        return COLORS["yolo_stable"]
    return COLORS["yolo"]


def _build_label(detection: dict) -> str:
    """Construct the display label for a detection."""
    parts = [
        f"#{detection['track_id']}",
        detection["final_label"],
        f"{detection['final_conf']:.2f}",
    ]

    swin_detail = detection.get("swin_detail") or detection.get("swin_label")
    if swin_detail:
        parts.append(f"[{swin_detail}]")

    swin_samples = detection.get("swin_samples", 0)
    if swin_samples > 0:
        parts.append(f"({swin_samples})")

    return " ".join(parts)


def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on a frame.

    Args:
        frame: BGR image (will be modified in-place).
        detections: List of detection dictionaries from VSHORADSystem.

    Returns:
        Annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _get_detection_color(det)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = _build_label(det)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame


def process_video(
    system: VSHORADSystem,
    video_path: str,
    output_dir: str,
    save_video: bool = True,
    save_json: bool = True,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Process a video file through the VSHORAD pipeline.

    Runs frame-by-frame detection, tracking, classification, and fusion.
    Optionally saves an annotated output video and a JSON results file.

    Args:
        system: Initialized VSHORADSystem instance.
        video_path: Path to input video file.
        output_dir: Directory for output files.
        save_video: Whether to write an annotated video.
        save_json: Whether to write per-frame JSON results.
        max_frames: Limit processing to N frames (None = full video).
        show_progress: Display a tqdm progress bar.

    Returns:
        Dictionary with processing statistics:
            - 'total_frames': Number of frames processed
            - 'avg_fps': Average processing speed
            - 'timing': Aggregated timing stats (mean YOLO/Swin/total ms)
            - 'output_video': Path to output video (if saved)
            - 'output_json': Path to JSON results (if saved)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    system.reset()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {video_path.name}, {W}x{H}, {fps:.1f} FPS, {total_frames} frames")

    # Output paths
    stem = video_path.stem
    output_video_path = output_dir / f"{stem}_processed.mp4"
    output_json_path = output_dir / f"{stem}_results.json"

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (W, H))

    all_results: List[dict] = []
    timing_stats = {"yolo": [], "swin": [], "total": []}

    iterator = range(total_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing")

    frame_idx = 0
    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        results = system.process_frame(frame, frame_idx)
        all_results.append(results)

        timing_stats["yolo"].append(results["timing"]["yolo_ms"])
        timing_stats["swin"].append(results["timing"]["swin_ms"])
        timing_stats["total"].append(results["timing"]["total_ms"])

        if writer is not None:
            annotated = draw_detections(frame, results["detections"])
            writer.write(annotated)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    # Save JSON results
    if save_json and all_results:
        # Make results JSON-serializable
        serializable = _make_serializable(all_results)
        with open(output_json_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Compute statistics
    stats = {
        "total_frames": frame_idx,
        "avg_fps": frame_idx / (sum(timing_stats["total"]) / 1000) if timing_stats["total"] else 0,
        "timing": {
            "yolo_ms_mean": float(np.mean(timing_stats["yolo"])) if timing_stats["yolo"] else 0,
            "swin_ms_mean": float(np.mean(timing_stats["swin"])) if timing_stats["swin"] else 0,
            "total_ms_mean": float(np.mean(timing_stats["total"])) if timing_stats["total"] else 0,
        },
        "tracking": system.stats.get("tracking", {}),
    }

    if save_video:
        stats["output_video"] = str(output_video_path)
        print(f"Saved video: {output_video_path}")
    if save_json:
        stats["output_json"] = str(output_json_path)
        print(f"Saved JSON: {output_json_path}")

    print(f"Done! {frame_idx} frames, {stats['avg_fps']:.1f} FPS avg")
    return stats


def _make_serializable(data: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(data, dict):
        return {k: _make_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_make_serializable(item) for item in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, tuple):
        return list(data)
    return data
