#!/usr/bin/env bash
# =============================================================================
# _train_yolo.sh  —  shared YOLO train → ONNX export → ONNX-load verify pipeline
#
# Called by scripts/train_yolov8s.sh and scripts/train_yolo11s.sh. Not run
# directly (but works if you do: _train_yolo.sh <model.pt> <run_name>).
#
# ASSUMPTIONS (override any via environment variable):
#   DATA_YAML   PLACEHOLDER default 'dataset/data.yaml' relative to repo root.
#               There is NO dataset in the repo — point this at a real YOLO
#               data.yaml before submitting. The script hard-fails if missing.
#   TRAIN_ENV   conda env for training. Default 'openyolo3d-dev'
#               (ultralytics 8.4.45 + yolo CLI; has NO onnx/onnxruntime).
#   EXPORT_ENV  conda env for export+verify. Default 'yolo-export'. Create once:
#                 conda create -n yolo-export python=3.11 -y
#                 conda activate yolo-export
#                 pip install ultralytics onnx onnxruntime-gpu==1.20.1 onnxslim
#   EPOCHS / BATCH / IMGSZ / CACHE  training knobs (defaults 100 / 32 / 1280 /
#               True). Override for a cheap smoke, e.g.
#                 EPOCHS=1 IMGSZ=320 CACHE=False bash scripts/train_yolov8s.sh
#   TRAIN_OVERRIDES  extra 'key=val' args appended to `yolo train`.
#
# CACHE=True loads images into RAM; on a large dataset this can exceed the job's
# memory — set CACHE=disk if training OOMs.
# =============================================================================
set -euo pipefail

MODEL="${1:?usage: _train_yolo.sh <model.pt> <run_name>}"   # yolov8s.pt | yolo11s.pt
RUN_NAME="${2:?usage: _train_yolo.sh <model.pt> <run_name>}"  # yolov8s | yolo11s

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_YAML="${DATA_YAML:-$REPO_ROOT/dataset/data.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-1280}"
CACHE="${CACHE:-True}"
TRAIN_ENV="${TRAIN_ENV:-openyolo3d-dev}"
EXPORT_ENV="${EXPORT_ENV:-yolo-export}"
PROJECT="$REPO_ROOT/runs/detect"
RUN_DIR="$PROJECT/$RUN_NAME"
LOG="$RUN_DIR/train.log"

mkdir -p "$RUN_DIR"

# --- conda ------------------------------------------------------------------
source /home/rintern16/miniconda3/etc/profile.d/conda.sh

# --- sanity -----------------------------------------------------------------
if [[ ! -f "$DATA_YAML" ]]; then
  echo "ERROR: DATA_YAML not found: $DATA_YAML" >&2
  echo "       Set DATA_YAML=/path/to/data.yaml (see header)." >&2
  exit 1
fi

echo "=== [$RUN_NAME] model=$MODEL data=$DATA_YAML ===" | tee -a "$LOG"

# --- train (openyolo3d-dev) -------------------------------------------------
conda activate "$TRAIN_ENV"
# pipefail keeps yolo's exit code even though we pipe through tee.
yolo detect train \
  model="$MODEL" \
  data="$DATA_YAML" \
  imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" \
  optimizer=AdamW lr0=0.001 weight_decay=0.0005 \
  patience=20 device=0 workers=8 \
  cache="$CACHE" amp=True cos_lr=True seed=42 \
  plots=True save_period=10 \
  project="$PROJECT" name="$RUN_NAME" exist_ok=True \
  ${TRAIN_OVERRIDES:-} 2>&1 | tee -a "$LOG"

BEST="$RUN_DIR/weights/best.pt"
[[ -f "$BEST" ]] || { echo "ERROR: best.pt not produced at $BEST" >&2; exit 1; }

# --- export + verify (yolo-export: has onnx + onnxruntime) ------------------
conda activate "$EXPORT_ENV"
yolo export model="$BEST" format=onnx imgsz="$IMGSZ" opset=17 simplify=True 2>&1 | tee -a "$LOG"

BEST_ONNX="$RUN_DIR/weights/best.onnx"
# Verify by running a real forward pass on a dummy input (not just loading).
python - "$BEST_ONNX" "$IMGSZ" <<'PY' 2>&1 | tee -a "$LOG"
import sys
import numpy as np
import onnxruntime as ort

path, imgsz = sys.argv[1], int(sys.argv[2])
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]

# Resolve dynamic dims: [batch->1, channels->3, H/W->imgsz].
shape = []
for i, d in enumerate(inp.shape):
    if isinstance(d, int) and d > 0:
        shape.append(d)
    else:
        shape.append(1 if i == 0 else 3 if i == 1 else imgsz)

dummy = np.zeros(shape, dtype=np.float32)
outs = sess.run(None, {inp.name: dummy})
print(f"ONNX inference OK: {path}")
print(f"  input  {inp.name} {shape}")
for o, arr in zip(sess.get_outputs(), outs):
    print(f"  output {o.name} {tuple(arr.shape)}")
PY

echo "=== [$RUN_NAME] DONE — artifacts in $RUN_DIR ===" | tee -a "$LOG"
