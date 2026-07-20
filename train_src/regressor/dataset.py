"""
dataset.py -- B 파이프라인의 regress_crop() 전처리와 완전히 동일해야 함.

B 코드 (pipeline_dongwook.py, regress_crop 함수) 기준:
    x1,y1 = max(0,x-5), max(0,y-5)
    x2,y2 = min(fw,x+w+5), min(fh,y+h+5)
    crop = frame[y1:y2, x1:x2]                      # BGR, cv2로 읽은 원본 그대로
    img = cv2.resize(crop,(128,128)).astype(np.float32)/255.0   # RGB 변환 없음!
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    meta = torch.tensor([[w/fw, h/fh, (y+h/2)/fh, focal_mm/40.0]])

⚠️ 핵심 주의사항 (반드시 지켜야 함):
  1. cv2로 이미지를 열어야 합니다 (PIL 사용 금지). PIL은 자동으로 RGB로
     변환하는데, B의 추론 코드는 BGR 그대로 넣기 때문에 PIL로 학습하면
     R/B 채널이 뒤바뀐 채로 학습 -> 실제 추론 때 색이 반대로 들어가서
     조용히 성능이 떨어지는 버그가 생깁니다.
  2. 이미지 정규화는 /255.0 뿐입니다. ImageNet mean/std 정규화를 넣으면
     학습-추론 전처리가 달라져서 안 됩니다.
  3. crop 이미지 저장 시 이미 bbox에 +-5px 패딩이 적용된 상태로
     저장되어 있어야 합니다 (Dataset Generator 쪽에서 크롭할 때
     동일한 5px 패딩을 적용해야 함 -- 이 부분은 B(데이터셋 생성) 담당자와
     반드시 확인 필요).

기대하는 labels.json 스키마 (예시):
{
  "samples": [
    {
      "image": "images/000001.png",     # 이미 +-5px 패딩 크롭된 이미지
      "bbox": [x, y, w, h],              # 패딩 "전" 원본 bbox (메타데이터 계산용)
      "image_width": 1920.0,
      "image_height": 1080.0,
      "camera": {"focal_length_mm": 5.14},
      "target": {"w": 30.5, "d": 20.8, "h": 15.6}
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, root_dir: str, labels_file: str = "labels.json", image_size: int = 128):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        with open(self.root_dir / labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_bgr(self, rel_path: str) -> np.ndarray:
        img_path = self.root_dir / rel_path
        # cv2.imread(str)는 Windows에서 비-ASCII 경로(한글 사용자명 등)를 못 여는
        # 경우가 있어 np.fromfile + cv2.imdecode로 우회 (디코딩 결과는 imread와 동일)
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, PIL 사용 안 함
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {img_path}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0  # B 코드와 동일: /255.0만, mean/std 정규화 없음
        return img.transpose(2, 0, 1)  # HWC -> CHW, 채널 순서는 BGR 그대로 유지

    def _build_metadata(self, sample: dict) -> np.ndarray:
        x, y, w, h = sample["bbox"]
        fw = sample["image_width"]
        fh = sample["image_height"]
        focal_mm = sample["camera"]["focal_length_mm"]
        # B regress_crop()과 정확히 동일한 4개 값, 동일한 순서
        return np.array(
            [w / fw, h / fh, (y + h / 2) / fh, focal_mm / 40.0],
            dtype=np.float32,
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = self._load_image_bgr(sample["image"])
        metadata = self._build_metadata(sample)

        target = sample["target"]
        target_cm = np.array([target["w"], target["d"], target["h"]], dtype=np.float32)
        log_target = np.log(np.clip(target_cm, 1e-3, None))  # 0 이하 값 방어

        return {
            "image": torch.from_numpy(image),
            "metadata": torch.from_numpy(metadata),
            "target": torch.from_numpy(log_target),
            "target_cm": torch.from_numpy(target_cm),
        }


def build_dataloaders(cfg: dict):
    """index 기반 train/val split (augmentation 없음 -- B 코드와 동일하게 유지)."""
    import random
    from torch.utils.data import DataLoader, Subset

    full = RegressionDataset(
        root_dir=cfg["root_dir"],
        labels_file=cfg.get("labels_file", "labels.json"),
        image_size=cfg.get("image_size", 128),
    )

    n = len(full)
    indices = list(range(n))
    random.Random(cfg.get("seed", 42)).shuffle(indices)
    val_len = max(1, int(n * cfg.get("val_ratio", 0.15)))
    val_idx = indices[:val_len]
    train_idx = indices[val_len:]

    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=cfg.get("batch_size", 16), shuffle=True,
        num_workers=cfg.get("num_workers", 2), drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.get("batch_size", 16), shuffle=False,
        num_workers=cfg.get("num_workers", 2),
    )
    return train_loader, val_loader


# 검증용 더미 데이터 생성기 (B 스키마 기준)
def create_dummy_data(root="dummy_dataset_dongwook", n=20):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(1, n + 1):
        img = np.random.randint(0, 255, (70, 60, 3), dtype=np.uint8)  # 5px 패딩 포함 크롭 가정
        cv2.imwrite(str(root / "images" / f"{i:06d}.png"), img)
        samples.append({
            "image": f"images/{i:06d}.png",
            "bbox": [100, 150, 50, 60],
            "image_width": 1920.0,
            "image_height": 1080.0,
            "camera": {"focal_length_mm": 5.14},
            "target": {"w": 30.0 + i * 0.5, "d": 20.0 + i * 0.3, "h": 15.0 + i * 0.2},
        })

    with open(root / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)
    return str(root)


if __name__ == "__main__":
    root = create_dummy_data()
    ds = RegressionDataset(root)
    sample = ds[0]
    print("image shape:", sample["image"].shape)      # (3, 128, 128) 이어야 함
    print("metadata:", sample["metadata"])              # 길이 4
    print("target(log):", sample["target"])
    assert sample["image"].shape == (3, 128, 128)
    assert sample["metadata"].shape == (4,)
    print("dataset.py 구조 검증 OK")
