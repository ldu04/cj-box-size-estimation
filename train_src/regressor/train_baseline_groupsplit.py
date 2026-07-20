# train_baseline_groupsplit.py — 기존(from-scratch 4-conv CNN) 아키텍처를
# train_backbone.py와 동일한 조건(그룹 분할·증강·박스단위 지표)으로 재학습.
#
# 백본 실험의 대조군: 데이터 소스(원본 224 크롭), 분할, 증강, 검증 프로토콜을
# 전부 맞추고 아키텍처(+입력 해상도 128, 기존 배포 규약)만 다르게 해서
# "백본 교체 자체"의 효과를 분리 측정한다.
# (기존 5.04cm는 크롭 단위 랜덤 분할(누수 가능)이라 직접 비교 불가)
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from model import build_model
from dataset224 import Crop224Dataset, group_split_indices, compute_size_balance_weights
from train_backbone import train_epoch, force_const_focal

_PERMS3 = list(itertools.permutations(range(3)))


def make_loss_fn(perm_invariant: bool, cm_space: bool):
    """순열 불변·cm공간 손실을 조합 가능하게 구성.
    cm_space=True면 log 파라미터화는 유지(수치 안정성)하되, 손실 자체는
    exp() 통과시켜 실제 cm 오차(공식 채점 단위)에서 직접 계산한다 — log 공간은
    큰 박스의 절대오차를 과소평가하는 구조적 불일치(40cm의 10%=4cm vs
    10cm의 10%=1cm를 동일 취급)가 있어 채점 기준과 안 맞는다."""
    def loss_fn(pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        if cm_space:
            pred, target = torch.exp(pred_log), torch.exp(target_log)
        else:
            pred, target = pred_log, target_log
        if perm_invariant:
            per_perm = torch.stack([
                F.smooth_l1_loss(pred[:, perm], target, reduction="none").sum(dim=1)
                for perm in _PERMS3
            ], dim=1)  # (B, 6)
            return per_perm.min(dim=1).values.mean()
        return F.smooth_l1_loss(pred, target)
    return loss_fn


@torch.no_grad()
def eval_epoch_permaware(model, loader, device, group_keys):
    """box_mae를 공식 채점과 동일하게 축 순열 최소 오차로 계산.
    (crop_mae는 참고용으로 순서 고정 그대로 유지 — 크롭 단위엔 순열 개념이
    적용되지 않고, 박스 단위 최종 예측에서만 순열 자유를 적용하는 것이 맞음)"""
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
        gt = gt_by_group[key]
        best = min(np.abs(med[list(perm)] - gt).mean() for perm in _PERMS3)
        box_errs.append(best)
    box_mae = float(np.mean(box_errs))
    return {"crop_mae": crop_mae, "box_mae": box_mae, "n_boxes": len(by_group)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True,
                   help="regressor_dataset 경로 (samples_224.json + crops_224/)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, default="regressor_baseline_groupsplit.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="out_dir/<ckpt_name>.last.pt가 있으면 이어서 학습 (Colab 연결 끊김 대비)")
    p.add_argument("--perm_invariant_loss", action="store_true",
                   help="w/d/h 순서 고정 SmoothL1 대신 6축 순열 중 최소 오차로 학습 "
                        "(공식 채점의 순열 자유와 목표를 일치시킴)")
    p.add_argument("--cm_space_loss", action="store_true",
                   help="손실을 log(cm)이 아니라 cm 공간에서 직접 계산 "
                        "(공식 채점이 절대 cm 오차라 log공간과 구조적 불일치 있음)")
    p.add_argument("--image_size", type=int, default=128,
                   help="크롭 리사이즈 해상도 (기존 배포 규약은 128; 224는 원본 그대로)")
    p.add_argument("--balance_by_size", action="store_true",
                   help="학습 샘플링을 GT max(w,d,h) 10cm bin의 역-빈도(sqrt, 상한 8x)로 "
                        "가중치를 줘서 큰 박스 표본을 오버샘플링 (2026-07-15 corr 분석: "
                        "GT w>=40cm가 전체의 12%뿐이라 큰 박스에서 예측이 포화되는 문제 대응). "
                        "val은 그대로 자연분포 유지(배포조건 재현).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, n_groups = group_split_indices(
        args.root_dir, args.val_ratio, args.split_seed
    )
    # 기존 아키텍처의 배포 규약은 128 입력 — --image_size로 224(원본 그대로) 실험 가능.
    # AdaptiveAvgPool을 쓰는 구조라 아키텍처 수정 없이 입력 해상도만 바뀜.
    train_ds = Crop224Dataset(args.root_dir, image_size=args.image_size,
                              augment=not args.no_augment)
    val_ds = Crop224Dataset(args.root_dir, image_size=args.image_size, augment=False)
    val_group_keys = [
        (val_ds.samples[i]["video"], val_ds.samples[i]["box_id"]) for i in val_idx
    ]

    if args.balance_by_size:
        weights = compute_size_balance_weights(args.root_dir, train_idx)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
        )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[BASELINE-control] groups={n_groups} train={len(train_idx)} val={len(val_idx)} "
          f"augment={not args.no_augment} image_size={args.image_size} "
          f"perm_invariant_loss={args.perm_invariant_loss} cm_space_loss={args.cm_space_loss} "
          f"balance_by_size={args.balance_by_size}")

    model = build_model().to(device)
    loss_fn = make_loss_fn(args.perm_invariant_loss, args.cm_space_loss)
    eval_fn = eval_epoch_permaware  # box_mae는 항상 순열 인식 방식으로 계산 (공식 채점과 일치)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_box_mae = float("inf")
    ckpt_path = out_dir / args.ckpt_name
    last_path = out_dir / f"{args.ckpt_name}.last.pt"
    history_path = out_dir / f"history_{args.ckpt_name}.json"
    history = []
    start_epoch = 1

    # 중단 후 재개: 모델+옵티마이저+스케줄러+에폭+best_box_mae+history 전부 복원
    if args.resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_box_mae = ckpt["best_box_mae"]
        start_epoch = ckpt["epoch"] + 1
        if history_path.is_file():
            with open(history_path) as f:
                history = json.load(f)
        print(f"[resume] epoch {start_epoch}부터 재개 (best_box_mae={best_box_mae:.2f}cm)")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, loss_fn, device)
        va = eval_fn(model, val_loader, device, val_group_keys)
        scheduler.step()
        print(f"[{epoch:03d}/{args.epochs}] train_mae={tr['mae']:.2f} | "
              f"val crop_mae={va['crop_mae']:.2f} box_mae={va['box_mae']:.2f}cm "
              f"({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train": tr, "val": va})
        if va["box_mae"] < best_box_mae:
            best_box_mae = va["box_mae"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best updated (box_mae {best_box_mae:.2f}cm), saved {ckpt_path}")

        # 매 에폭 즉시 저장 (재개용 체크포인트 + 학습 로그) — 연결이 끊겨도
        # 이 시점까지는 보존되고 --resume으로 이어서 돌릴 수 있음
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_box_mae": best_box_mae,
        }, last_path)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"DONE. [BASELINE-control] best val box-level MAE = {best_box_mae:.2f}cm")


if __name__ == "__main__":
    main()
