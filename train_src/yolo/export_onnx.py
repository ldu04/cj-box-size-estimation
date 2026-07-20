#!/usr/bin/env python3
"""Export a trained YOLOv8/YOLO11 best.pt to checkpoints/model.onnx and verify it.

Usage:
    python scripts/export_onnx.py runs/detect/yolov8s/weights/best.pt
    python scripts/export_onnx.py <best.pt> [--out checkpoints/model.onnx]

Fixed export params per the CJ challenge eval env (ORT 1.20.1, opset 17):
    imgsz=1280  opset=17  simplify=True  dynamic=False

Verification: onnx.checker -> onnxruntime.InferenceSession -> one dummy forward.
Exits non-zero if export or any validation step fails.
Run in the `yolo-export` conda env (has ultralytics + onnx + onnxruntime).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

IMGSZ = 1280
OPSET = 17


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", help="path to trained best.pt (YOLOv8 or YOLO11)")
    ap.add_argument("--out", default="checkpoints/model.onnx",
                    help="destination .onnx path (default: checkpoints/model.onnx)")
    args = ap.parse_args()

    pt = Path(args.weights)
    if not pt.is_file():
        print(f"ERROR: weights not found: {pt}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- export (ultralytics writes best.onnx next to the .pt) ---------------
    print(f"=== exporting {pt} -> {out} (imgsz={IMGSZ} opset={OPSET}) ===")
    produced = Path(YOLO(str(pt)).export(
        format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False,
    ))
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))
    print(f"exported: {out}")

    # --- verify 1: static graph check ----------------------------------------
    onnx.checker.check_model(onnx.load(str(out)))
    print("onnx.checker: OK")

    # --- verify 2: load + one dummy forward ----------------------------------
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # dynamic=False so dims are concrete, but resolve defensively anyway.
    shape = [d if isinstance(d, int) and d > 0
             else (1 if i == 0 else 3 if i == 1 else IMGSZ)
             for i, d in enumerate(inp.shape)]
    outs = sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})

    print("onnxruntime inference: OK")
    print(f"  input  {inp.name} {shape}")
    for o, arr in zip(sess.get_outputs(), outs):
        print(f"  output {o.name} {tuple(arr.shape)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: export/validation failed: {e}", file=sys.stderr)
        sys.exit(1)
