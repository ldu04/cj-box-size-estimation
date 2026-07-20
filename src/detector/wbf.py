"""Weighted Box Fusion (WBF) — merges detections from multiple independently
trained models into one consensus set, weighting each model's contribution.

Pure numpy, no external deps (must run in the eval env with no extra pip
installs). Standalone and unit-testable in isolation from any ONNX model.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    weights: List[float],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    boxes_list[i]: (N_i, 4) xyxy boxes from model i (same pixel coord frame).
    scores_list[i]: (N_i,) confidence scores from model i.
    weights[i]: reliability weight for model i.

    Clusters boxes across all models by IoU, fuses each cluster into one box
    (score-and-weight-weighted centroid), and scales the fused score by
    agreement (how many of the N models contributed to the cluster) — a
    box only one model saw scores lower than one every model agrees on.

    Returns (fused_boxes (M,4), fused_scores (M,)).
    """
    n_models = len(boxes_list)
    all_boxes, all_scores, all_weights = [], [], []
    for boxes, scores, w in zip(boxes_list, scores_list, weights):
        for b, s in zip(boxes, scores):
            if s < skip_box_thr:
                continue
            all_boxes.append(b)
            all_scores.append(s)
            all_weights.append(w)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,))

    all_boxes = np.asarray(all_boxes, dtype=np.float64)
    all_scores = np.asarray(all_scores, dtype=np.float64)
    all_weights = np.asarray(all_weights, dtype=np.float64)

    order = np.argsort(-all_scores)
    used = np.zeros(len(order), dtype=bool)

    fused_boxes, fused_scores = [], []
    for pos in range(len(order)):
        idx = order[pos]
        if used[idx]:
            continue
        remaining = order[pos + 1:]
        remaining = remaining[~used[remaining]]
        if remaining.size:
            ious = _iou_one_to_many(all_boxes[idx], all_boxes[remaining])
            cluster = np.concatenate([[idx], remaining[ious >= iou_thr]])
        else:
            cluster = np.array([idx])
        used[cluster] = True

        member_weight = all_weights[cluster] * all_scores[cluster]
        fused_box = (all_boxes[cluster] * member_weight[:, None]).sum(axis=0) / member_weight.sum()
        fused_score = (all_scores[cluster] * all_weights[cluster]).sum() / all_weights[cluster].sum()
        fused_score *= min(len(cluster), n_models) / n_models

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)

    return np.array(fused_boxes), np.array(fused_scores)
