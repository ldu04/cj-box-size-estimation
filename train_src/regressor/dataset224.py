"""
dataset224.py — 원본 224px 크롭(regressor_dataset/samples_224.json)을 직접 읽는
Dataset + 라벨-안전 증강.

기존 dataset.py(regression_dataset, 128px 축소본)와의 차이:
- 원본 224 해상도 그대로 사용 (128로 뭉갠 뒤 업샘플하는 화질 손실 제거)
- 증강 3종 (train 전용, 모두 라벨 불변이 보장되는 것만):
    * 좌우 반전 — w/d/h와 meta(bbox w/fw, h/fh, cy/fh) 모두 반전 불변
    * 밝기/대비 지터 — hidden test의 "조명 변화" 대응
    * focal 지터(상대 ±3%) — 추론 시 FocalNet 추정치(오차 ~3.2%)가 들어가는
      조건을 학습에서 미리 노출
- (video, box_id) 그룹 단위 분할 헬퍼 — 같은 박스의 크롭이 train/val에
  동시에 들어가는 누수 차단
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Crop224Dataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 224, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        with open(self.root_dir / "samples_224.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data["samples"] if isinstance(data, dict) else data

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_bgr(self, crop_name: str) -> np.ndarray:
        img_path = self.root_dir / "crops_224" / crop_name
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {img_path}")
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_image_bgr(s["crop"])

        x, y, w, h = s["bbox_px"]
        fw, fh = s["frame_res"]
        focal_mm = s["camera"]["focal_length_mm"]

        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1]  # 좌우 반전 (라벨·meta 불변)
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)   # 대비
                beta = random.uniform(-20, 20)       # 밝기
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
            if random.random() < 0.5:
                focal_mm = focal_mm * random.uniform(0.97, 1.03)  # focalnet 오차 모사

        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC(BGR) -> CHW, BGR 유지 (기존 규약)

        metadata = np.array(
            [w / fw, h / fh, (y + h / 2) / fh, focal_mm / 40.0],
            dtype=np.float32,
        )
        target_cm = np.array([s["w"], s["d"], s["h"]], dtype=np.float32)
        log_target = np.log(np.clip(target_cm, 1e-3, None))

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "metadata": torch.from_numpy(metadata),
            "target": torch.from_numpy(log_target),
            "target_cm": torch.from_numpy(target_cm),
        }


def group_split_indices(root_dir: str, val_ratio: float, split_seed: int):
    """(video, box_id) 단위로 나눈 (train_idx, val_idx, n_groups) 반환."""
    with open(Path(root_dir) / "samples_224.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data

    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[(s["video"], s["box_id"])].append(i)
    group_list = list(groups.values())
    random.Random(split_seed).shuffle(group_list)

    n_val = max(1, int(len(group_list) * val_ratio))
    val_idx = [i for g in group_list[:n_val] for i in g]
    train_idx = [i for g in group_list[n_val:] for i in g]
    return train_idx, val_idx, len(group_list)


def compute_size_balance_weights(
    root_dir: str, indices: list[int], bin_size: float = 10.0, max_weight_ratio: float = 8.0,
) -> list[float]:
    """indices(보통 train_idx)에 대해, GT max(w,d,h)를 bin_size cm 단위로 나눠
    표본이 적은(=큰 박스) bin일수록 가중치를 높게 준다 (역-빈도, sqrt로 완화).

    2026-07-15 corr 분석(corr(gt_w, error_w)=-0.885, GT w>=40cm가 전체의 12%뿐)
    에서 나온 가설 — 손실함수(순열불변+cm공간)를 바꿔도 안 고쳐진 큰 박스
    과소예측 편향이, 학습데이터의 큰 박스 표본 부족 때문일 수 있다는 것.
    특정 영상을 겨냥하지 않고 GT 크기라는 일반 속성으로만 가중치를 매겨
    오버피팅 방지 원칙(특정 문제 영상 손튜닝 금지)을 지킨다.

    max_weight_ratio: 최고/최저 가중치 비율 상한 — 표본이 극소수(n<10)인
    극단 bin이 배치를 지배해 그 몇 개 크롭에 과적합하는 것을 방지.
    """
    with open(Path(root_dir) / "samples_224.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data

    max_dims = [max(samples[i]["w"], samples[i]["d"], samples[i]["h"]) for i in indices]
    bin_ids = [int(d // bin_size) for d in max_dims]
    bin_counts = defaultdict(int)
    for b in bin_ids:
        bin_counts[b] += 1

    raw_weights = [1.0 / (bin_counts[b] ** 0.5) for b in bin_ids]
    lo, hi = min(raw_weights), max(raw_weights)
    capped_hi = lo * max_weight_ratio
    weights = [min(w, capped_hi) for w in raw_weights]
    return weights
