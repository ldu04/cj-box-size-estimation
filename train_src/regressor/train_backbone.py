# train_backbone.py — MobileNetV3-Small 백본 회귀 모델 학습 (실험).
#
# 기존 train_variant.py 대비 바뀐 점 (사전 검토에서 확정한 것들):
# 1. 원본 224px 크롭(regressor_dataset) 직접 사용 — 기존은 128 축소본
#    (regression_dataset)을 썼고, 백본에 넣으려고 다시 업샘플하면 화질 손실.
# 2. (video, box_id) 그룹 단위 분할 — 크롭 단위 랜덤 분할은 같은 박스의 다른
#    프레임이 train/val 양쪽에 들어가는 누수 위험 (3418 크롭 / 885 박스).
# 3. 라벨-안전 증강 3종 (dataset224.py): h-flip, 밝기/대비, focal 지터.
#    hidden test의 "조명 변화" 명시에 직접 대응.
# 4. 검증을 크롭 단위 + **박스 단위(그룹 중앙값)** 둘 다 리포트 — 배포
#    파이프라인은 트랙당 크롭들의 중앙값을 쓰므로 박스 단위가 실전 지표.
# 5. F-const 프로토콜 유지: val은 항상 상수 focal로 채점 (실전 조건).
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from model_backbone import build_model
from dataset224 import Crop224Dataset, group_split_indices

CONST_FOCAL = 8.767


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True,
                   help="regressor_dataset 경로 (samples_224.json + crops_224/)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, default="regressor_backbone.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--no_augment", action="store_true")
    return p.parse_args()


def force_const_focal(meta: torch.Tensor) -> torch.Tensor:
    meta = meta.clone()
    meta[:, 3] = CONST_FOCAL / 40.0
    return meta


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_mae = 0.0
    n = 0
    for batch in loader:
        img = batch["image"].to(device)
        meta = batch["metadata"].to(device)
        target_log = batch["target"].to(device)
        target_cm = batch["target_cm"].to(device)
        optimizer.zero_grad()
        pred_log = model(img, meta)
        loss = loss_fn(pred_log, target_log)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            total_mae += torch.mean(torch.abs(torch.exp(pred_log) - target_cm)).item()
        n += 1
    n = max(n, 1)
    return {"loss": total_loss / n, "mae": total_mae / n}


@torch.no_grad()
def eval_epoch(model, loader, device, group_keys):
    """crop 단위 MAE + 박스 단위(그룹 중앙값, 배포 방식 모사) MAE 둘 다 계산."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        img = batch["image"].to(device)
        meta = force_const_focal(batch["metadata"]).to(device)
        pred_log = model(img, meta)
        preds.append(torch.exp(pred_log).cpu().numpy())
        targets.append(batch["target_cm"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    crop_mae = float(np.mean(np.abs(preds - targets)))

    by_group = defaultdict(list)
    gt_by_group = {}
    for key, p, t in zip(group_keys, preds, targets):
        by_group[key].append(p)
        gt_by_group[key] = t
    box_errs = []
    for key, plist in by_group.items():
        med = np.median(np.stack(plist), axis=0)
        box_errs.append(np.abs(med - gt_by_group[key]))
    box_mae = float(np.mean(np.stack(box_errs)))
    return {"crop_mae": crop_mae, "box_mae": box_mae, "n_boxes": len(by_group)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, n_groups = group_split_indices(
        args.root_dir, args.val_ratio, args.split_seed
    )
    train_ds = Crop224Dataset(args.root_dir, image_size=args.image_size,
                              augment=not args.no_augment)
    val_ds = Crop224Dataset(args.root_dir, image_size=args.image_size, augment=False)
    val_group_keys = [
        (val_ds.samples[i]["video"], val_ds.samples[i]["box_id"]) for i in val_idx
    ]

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"groups(boxes)={n_groups} train_crops={len(train_idx)} val_crops={len(val_idx)} "
          f"augment={not args.no_augment} image_size={args.image_size}")
    print("(group-level split / val: const-focal, crop+box-level metrics)")

    model = build_model().to(device)
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = AdamW(
        model.param_groups(args.backbone_lr, args.head_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_box_mae = float("inf")
    ckpt_path = out_dir / args.ckpt_name
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, loss_fn, device)
        va = eval_epoch(model, val_loader, device, val_group_keys)
        scheduler.step()
        print(f"[{epoch:03d}/{args.epochs}] train_mae={tr['mae']:.2f} | "
              f"val crop_mae={va['crop_mae']:.2f} box_mae={va['box_mae']:.2f}cm "
              f"({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train": tr, "val": va})
        if va["box_mae"] < best_box_mae:
            best_box_mae = va["box_mae"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best updated (box_mae {best_box_mae:.2f}cm), saved {ckpt_path}")

    with open(out_dir / f"history_{args.ckpt_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"DONE. best val box-level MAE(const-focal, group split) = {best_box_mae:.2f}cm")
    print("박스 단위(그룹 중앙값) 지표가 배포 파이프라인 성능에 직결되는 값입니다.")


if __name__ == "__main__":
    main()
