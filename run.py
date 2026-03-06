#!/usr/bin/env python3
"""
VSHORAD Aircraft Detection System — CLI entry point.

Usage:
    python run.py --tier strategic --video test.mp4
    python run.py --tier tactical --video test.mp4 --output results/
    python run.py --tier embedded --video test.mp4 --max-frames 300

Requires pre-trained weights in the weights/ directory.
"""

import argparse
from pathlib import Path

from src.config import Tier
from src.system import VSHORADSystem
from src.video import process_video


# Default weight paths per tier
DEFAULT_WEIGHTS = {
    Tier.STRATEGIC: {
        "yolo": "weights/strategic/yolov8l_1280_best.pt",
        "swin": "weights/strategic/swin_base_384_best.pth",
    },
    Tier.TACTICAL: {
        "yolo": "weights/tactical/yolov8m_960_best.pt",
        "swin": "weights/tactical/swin_small_224_best.pth",
    },
    Tier.EMBEDDED: {
        "yolo": "weights/embedded/yolov8m_640_fp16.engine",
        "swin": "weights/embedded/swin_small_224_fp16.engine",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="VSHORAD Aircraft Detection & Classification System"
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["strategic", "tactical", "embedded"],
        default="strategic",
        help="System tier (default: strategic)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLO weights (overrides tier default)",
    )
    parser.add_argument(
        "--swin-weights",
        type=str,
        default=None,
        help="Path to Swin weights (overrides tier default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit processing to N frames",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing output video",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON results",
    )
    args = parser.parse_args()

    tier = Tier(args.tier)
    defaults = DEFAULT_WEIGHTS[tier]

    yolo_path = args.yolo_weights or defaults["yolo"]
    swin_path = args.swin_weights or defaults["swin"]

    # Validate paths
    for name, path in [("YOLO weights", yolo_path), ("Swin weights", swin_path)]:
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}")
            print(f"Download weights and place them in the expected location.")
            print(f"See README.md for instructions.")
            return

    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        return

    # Initialize system
    system = VSHORADSystem(
        yolo_weights=yolo_path,
        swin_weights=swin_path,
        tier=tier,
        device=args.device,
    )

    # Process video
    stats = process_video(
        system=system,
        video_path=args.video,
        output_dir=args.output,
        save_video=not args.no_video,
        save_json=not args.no_json,
        max_frames=args.max_frames,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Tier:         {tier.value}")
    print(f"  Frames:       {stats['total_frames']}")
    print(f"  Avg FPS:      {stats['avg_fps']:.1f}")
    print(f"  YOLO (avg):   {stats['timing']['yolo_ms_mean']:.1f} ms")
    print(f"  Swin (avg):   {stats['timing']['swin_ms_mean']:.1f} ms")
    print(f"  Total (avg):  {stats['timing']['total_ms_mean']:.1f} ms")
    tracking = stats.get("tracking", {})
    if tracking:
        print(f"  Re-IDs:       {tracking.get('total_reids', 0)}")


if __name__ == "__main__":
    main()
