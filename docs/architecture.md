# System Architecture

> Detailed description of the VSHORAD pipeline: detection, tracking, classification, and sensor fusion.

![System Architecture](../assets/architecture_pipeline.png)

## Pipeline Overview

The system processes video frames through five stages:

1. **Frame Acquisition** — Input from an observation camera (1920×1080 or higher)
2. **Detection** — YOLOv8 localizes aerial objects and assigns one of 12 meta-categories
3. **Tracking** — ByteTrack assigns persistent track IDs across frames using Kalman filtering
4. **Classification** — Swin Transformer identifies the specific aircraft type from 56 classes
5. **Feedback Loop** — EMA aggregation, stability checks, Re-ID, and unidentified detection

## Detection Module (YOLOv8)

The detection module uses the YOLOv8 anchor-free architecture from Ultralytics. YOLOv8 processes the entire image in a single forward pass, performing detection and classification simultaneously.

**Architecture components:**

- **Backbone (CSPDarknet)**: Extracts hierarchical feature maps by splitting and recombining feature channels through dense connections. Deeper layers capture increasingly abstract representations.
- **Neck (PANet + FPN)**: Fuses multi-scale features. FPN provides top-down semantic enrichment while PANet adds bottom-up localization precision. This enables detection of objects at varying scales.
- **Detection Head**: Anchor-free head that directly predicts object centers, widths, and heights across three detection scales (P3/P4/P5) for small, medium, and large objects respectively.

### Model Variants

| Tier | Variant | Resolution | Parameters | GFLOPs |
|------|---------|------------|------------|--------|
| Strategic | YOLOv8-Large | 1280×1280 | 43.7M | 165.2 |
| Tactical/Embedded | YOLOv8-Medium | 960×960 / 640×640 | 25.9M | 78.9 |

Higher input resolution directly improves small-object detection. At 1280px, a 20-pixel object occupies more feature map cells than at 640px, which is critical for detecting distant aircraft in VSHORAD scenarios.

### Inference Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Confidence threshold | 0.25 | Intentionally low — the Feedback Loop handles false positive filtering over time |
| IoU threshold (NMS) | 0.45 | Eliminates duplicate detections of the same object |
| Classes | 12 | Meta-categories: Fighter, Helicopter, Transport, Bomber, Special, UAV_Fixed, UAV_Rotor, Missile, Civilian, Birds, Decoys, Unidentified |
| Test-Time Augmentation | Disabled | Speed priority over marginal accuracy gain |

## Tracking Module (ByteTrack)

ByteTrack maintains persistent object identity across frames. Unlike traditional trackers that discard low-confidence detections, ByteTrack uses them to recover partially occluded or motion-blurred objects.

### Two-Stage Association

1. **High-confidence matching** (conf > 0.45): Associates strong detections with active tracks using the Hungarian algorithm, minimizing IoU-based assignment cost.
2. **Low-confidence recovery** (0.10 < conf < 0.45): Associates remaining weak detections with unmatched tracks — recovers objects that are temporarily hard to detect.

### Position Prediction

A Kalman filter maintains state for each track: center position, bounding box dimensions, and their velocities. This predicts where each object should appear in the next frame, enabling association even when objects move significantly between frames.

### Track Lifecycle

- **Creation**: New tracks require confidence > 0.55 (higher than association threshold to prevent false trajectories from single spurious detections)
- **Maintenance**: Tracks persist for up to 90 frames (~3 seconds at 30 FPS) without matched detections
- **Deletion**: Tracks without any detection for `track_buffer` frames are removed

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `track_buffer` | 90 | Frames before track deletion (~3s @ 30fps) |
| `match_thresh` | 0.92 | IoU threshold for association (high = more tolerant) |
| `track_high_thresh` | 0.45 | High-confidence detection threshold |
| `track_low_thresh` | 0.10 | Low-confidence detection threshold |
| `new_track_thresh` | 0.55 | Minimum confidence for new track creation |
| `fuse_score` | True | Fuse detection confidence with tracking score |

## Classification Module (Swin Transformer)

The classification module performs fine-grained identification using crops extracted from YOLO detections. While YOLO assigns broad meta-categories (e.g., "Fighter"), Swin identifies the specific type (e.g., "F-16").

### Shifted Window Attention

Standard Vision Transformers compute global attention across all image patches (O(N²) complexity). Swin Transformer computes attention locally within fixed-size windows, achieving linear complexity. Information flows between windows through alternating window positions (shifted windows) across layers.

The model has four stages, each at decreasing spatial resolution. Early stages extract local features (edges, textures), while deeper stages learn abstract patterns characteristic of specific aircraft types.

### Model Variants

| Tier | Variant | Resolution | Window | Parameters | GFLOPs |
|------|---------|------------|--------|------------|--------|
| Strategic | Swin-Base | 384×384 | 12×12 | 86.9M | 47.1 |
| Tactical/Embedded | Swin-Small | 224×224 | 7×7 | 48.9M | 8.7 |

Both models were pre-trained on ImageNet-22K (14M images, 21K categories) and fine-tuned on the Skydetect Balanced Swin Dataset (140,000 images, 56 classes).

### Preprocessing

Input crops go through: letterbox resize (preserving aspect ratio with black padding) → ImageNet normalization → softmax output producing a 56-dimensional probability vector.

## Feedback Loop

![Feedback Loop](../assets/feedback_loop.png)

The Feedback Loop is the key integration component, responsible for temporal aggregation, prediction stabilization, and track recovery.

### EMA Aggregation

For each tracked object, a 56-dimensional probability vector is maintained and updated via Exponential Moving Average:

```
P_t = α · P_swin + (1 - α) · P_{t-1}     where α = 0.4
```

This ensures single misclassifications don't immediately change the output, aggregates information from multiple observations, and provides robustness against momentary lighting or perspective changes.

### Selective Classification

Not all detections are sent to Swin — routing rules optimize throughput:

| YOLO Category | Classification Rule | Rationale |
|---------------|-------------------|-----------|
| Fighter, Bomber | **Always** classify | High operational priority |
| Helicopter, Transport, Special | Classify if YOLO conf < 0.70 | Only when YOLO is uncertain |
| UAV, Missile, Civilian, Birds, Decoys, Unidentified | **Never** classify | YOLO meta-category is sufficient |

### Re-Identification (Re-ID)

When a tracked object temporarily disappears (occlusion, camera movement), the system stores its last position, velocity, EMA probabilities, and YOLO class. When a new unmatched detection appears, it checks for potential re-identification:

```
predicted_position = last_position + velocity × frames_lost × 1.5
```

A match occurs when: distance < 200px, same YOLO class, and absence < 45 frames (~1.5s). Successful Re-ID restores the full EMA classification history, avoiding the need to rebuild confidence from scratch.

### Unidentified Detection

Tracks with excessive label oscillation are flagged as "Unidentified". If more than 10 class changes occur within a 30-frame window, the track is marked unstable. This prevents rapid label flickering for hard-to-classify objects (partially occluded, distant, or unusual geometry) and instead honestly communicates uncertainty to the operator.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `swin_ema_alpha` | 0.4 | EMA smoothing coefficient |
| `reclassify_interval` | 25 | Frames between Swin re-invocations |
| `reid_max_frames` | 45 | Maximum absence for Re-ID (~1.5s @ 30fps) |
| `reid_max_distance` | 200 | Maximum pixel distance for Re-ID matching |
| `reid_velocity_mult` | 1.5 | Position prediction uncertainty multiplier |
| `yolo_stable_frames` | 8 | Frames for YOLO class stabilization |
| `swin_override_threshold` | 0.80 | Swin confidence to override YOLO |
| `yolo_weak_threshold` | 0.50 | YOLO confidence below which Swin can override |
| `unidentified_window` | 30 | Analysis window for stability check |
| `unidentified_max_changes` | 10 | Label changes triggering Unidentified status |

## Deployment Tiers

| Parameter | Strategic | Tactical | Embedded |
|-----------|-----------|----------|----------|
| YOLO model | YOLOv8-Large | YOLOv8-Medium | YOLOv8-Medium |
| YOLO resolution | 1280px | 960px | 640px |
| Swin model | Swin-Base | Swin-Small | Swin-Small |
| Swin resolution | 384px | 224px | 224px |
| YOLO backend | PyTorch | PyTorch | TensorRT FP16 |
| Swin backend | PyTorch | PyTorch | ONNX Runtime |
| FPS (NVIDIA L4) | 24.5 | 41.2 | 47.0 |
| Speedup | 1.00× | 1.68× | 1.92× |
| Target | Command center | Mobile station | Edge / vehicle-mounted |

The Embedded tier achieved 47.0 FPS with PyTorch fallback due to TensorRT compatibility issues during testing. With native TensorRT FP16, a 50–70% speedup is expected (targeting 70–80+ FPS on Jetson AGX Orin).
