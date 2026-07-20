#!/usr/bin/env bash
# Experiment 2: YOLO11s. Trains, exports best.onnx, verifies ONNX-Runtime load.
# All config + assumptions (DATA_YAML placeholder, envs, knobs) live in
# scripts/_train_yolo.sh — read its header before running.
exec "$(dirname "${BASH_SOURCE[0]}")/_train_yolo.sh" yolo11s.pt yolo11s
