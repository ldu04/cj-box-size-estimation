# train_focal.py — 전체 프레임에서 focal(초점거리)을 추정하는 소형 CNN.
#
# 배경: 테스트 영상에는 카메라 메타데이터가 없어 regressor의 focal 입력이
# 상수로 고정된다(F-const). 프레임의 화각/원근 단서로 focal을 추정할 수
# 있다면, focal-aware regressor(정답 focal 학습본)와 조합해 상수 대비
# 정확도를 회복할 수 있다 (상한: 실측 4.67 vs 상수 5.04).
#
# 분할은 반드시 영상 단위 — 같은 영상의 프레임이 train/val에 갈리면 누수.
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FocalDataset(Dataset):
    def __init__(self, root: str, samples: list):
        self.root = Path(root)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        buf = np.fromfile(str(self.root / s["frame"]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(img.transpose(2, 0, 1)),   # (3,180,320) BGR
            "target": torch.tensor([np.log(s["focal_length_mm"])], dtype=torch.float32),
            "focal": torch.tensor([s["focal_length_mm"]], dtype=torch.float32),
        }


class FocalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, img):
        return self.head(self.cnn(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default=r"C:\Users\이동욱\Desktop\assignment1\focal_dataset")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../checkpoints/focalnet.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with open(Path(args.root_dir) / "labels.json", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    # 영상 단위 분할
    vids = sorted({s["video_id"] for s in samples})
    random.Random(args.seed).shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_ratio))
    val_vids = set(vids[:n_val])
    tr = [s for s in samples if s["video_id"] not in val_vids]
    va = [s for s in samples if s["video_id"] in val_vids]
    print(f"train frames={len(tr)} ({len(vids)-n_val} videos) / "
          f"val frames={len(va)} ({n_val} videos)")

    tl = DataLoader(FocalDataset(args.root_dir, tr), batch_size=args.batch_size,
                    shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(FocalDataset(args.root_dir, va), batch_size=args.batch_size,
                    shuffle=False, num_workers=0)

    model = FocalNet()
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for b in tl:
            opt.zero_grad()
            loss = loss_fn(model(b["image"]), b["target"])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        errs, rels = [], []
        with torch.no_grad():
            for b in vl:
                pred = torch.exp(model(b["image"]))
                errs.append((pred - b["focal"]).abs())
                rels.append(((pred - b["focal"]).abs() / b["focal"]))
        mae = torch.cat(errs).mean().item()
        rel = torch.cat(rels).mean().item()
        if mae < best:
            best = mae
            torch.save(model.state_dict(), args.out)
            flag = " *best*"
        else:
            flag = ""
        if ep % 10 == 0 or flag:
            print(f"[{ep:03d}/{args.epochs}] val focal MAE={mae:.3f}mm "
                  f"(rel {rel*100:.1f}%) ({time.time()-t0:.1f}s){flag}")

    print(f"DONE. best val focal MAE = {best:.3f}mm -> {args.out}")


if __name__ == "__main__":
    main()
