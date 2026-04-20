# Deployment Notes

This document covers weights provenance, the repository's origin as a conversion
from research notebooks, and the intentional scope of the Python inference layer.
It addresses practical questions that arise when trying to reproduce results on
hardware other than the original training/deployment targets.

## Weights Provenance

All training was performed on Google Colab GPU runtimes. The weights directory
contents have the following origin:

| Tier      | YOLO                          | Swin                                      | Training GPU | Format         |
|-----------|-------------------------------|-------------------------------------------|--------------|----------------|
| Strategic | `yolov8l_1280_best.pt`        | `swin_base_384_best.pth`                  | A100 40GB    | PyTorch        |
| Tactical  | `yolov8m_960_best.pt`         | `swin_small_224_best.pth`                 | L4 24GB      | PyTorch        |
| Embedded  | `yolov8m_640_fp16.engine`     | `swin_small_224_fp16.engine`              | L4 24GB      | TensorRT       |

Embedded tier engines were built from ONNX exports on Colab L4 (compute
capability 8.9, CUDA 12.1, TensorRT 8.6). They are **not portable** across
GPU architectures — TensorRT engines encode kernel selections and memory
layouts for the specific compute capability and runtime version used at
build time. An engine built on L4 will not deserialize on RTX 3060
(compute 8.6), Jetson Orin (compute 8.7), or H100 (compute 9.0). This is
documented TensorRT behavior.

To deploy the embedded tier on different hardware, either:

1. Run the end-to-end pipeline on the originally targeted Jetson Orin NX.
2. Rebuild engines on the target hardware using the ONNX export and TRT
   build steps in `training/03_export_embedded.ipynb`. This requires
   TensorRT SDK and CUDA toolkit installed locally.

## Repository Origin

This project was developed iteratively in Jupyter/Colab notebooks during
the thesis period. The `src/` Python package was refactored out of the
working notebooks after thesis defense for reproducibility and version
control. The conversion was deliberate: notebooks remain the canonical
training environment, and `src/` is the deployment runtime.

As a consequence, some components exist only in notebook form:

- **ONNX export pipeline** — `training/03_export_embedded.ipynb`
- **TensorRT engine build configuration** — same notebook
- **Latency benchmarking and tier comparison** — `notebooks/eval_latency.ipynb`

This split is intentional. The notebooks are reproducible environments
tied to specific Colab runtime versions and are not meant for deployment.
The `src/` package is structured for inference-time use on target
hardware.

## Python Inference Layer Scope

The Python inference classes (`YOLODetector`, `SwinClassifier`) use
standard PyTorch loaders and are **not currently compatible with
TensorRT engines**. Attempting to load `.engine` files through these
classes produces:

- `torch.load()` on `.engine` → `_pickle.UnpicklingError: invalid load
  key, 'f'` (the first byte of a TRT engine is `f` from its `ftrt`
  signature, not a pickle magic number)
- `ultralytics.YOLO()` on `.engine` → **silent fallback** to downloading
  generic COCO weights from Ultralytics hub. The pipeline appears to
  initialize but runs on wrong classes. This is a known ultralytics
  behavior worth guarding against in production deployment with an
  explicit class-name validation step after model load.

The embedded tier was originally intended to run on Jetson with a
separate TRT-based loader outside this repository. Adding `TrtYOLODetector`
and `TrtSwinClassifier` to the Python inference layer is on the
post-thesis roadmap. Estimated effort: 2–3 days including ONNX export
verification, TRT runtime integration, and engine-format detection
dispatch in `run.py`.

Until then, Strategic and Tactical tiers are the runnable configurations
on non-Jetson hardware.