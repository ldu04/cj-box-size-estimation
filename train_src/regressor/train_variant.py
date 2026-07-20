# train_variant.py — focal 처리 방식 실험용 학습 스크립트.
# model.py / dataset.py / train.py(산출물)는 건드리지 않고 재사용한다.
#
# --focal_mode:
#   true   : 정답 focal 사용 (기존 train.py와 동일)
#   const  : focal을 항상 CONST_FOCAL(8.767)로 고정 — 실전(테스트) 조건과 동일
#   jitter : 배치마다 50% 확률로 정답/상수를 섞음
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_model
from dataset import build_dataloaders

CONST_FOCAL = 8.767  # 학습 크롭 3418개의 focal 중앙값 (mm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--focal_mode", choices=["true", "const", "jitter"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


def apply_focal_mode(meta: torch.Tensor, mode: str, train: bool) -> torch.Tensor:
    """meta[:,3] = focal/40 채널을 focal_mode에 맞게 변환.

    검증 시에는 항상 실전 조건(const)으로 평가한다 — 테스트 영상에는
    focal 메타데이터가 없어 상수가 들어가기 때문.
    """
    if mode == "true" and train:
        return meta
    meta = meta.clone()
    if mode == "jitter" and train:
        mask = torch.rand(meta.shape[0]) < 0.5
        meta[mask, 3] = CONST_FOCAL / 40.0
        return meta
    # const 학습, 그리고 모든 모드의 검증: 상수 고정
    meta[:, 3] = CONST_FOCAL / 40.0
    return meta


@torch.no_grad()
def mae_cm(pred_log, target_cm):
    return torch.mean(torch.abs(torch.exp(pred_log) - target_cm)).item()


def run_epoch(model, loader, optimizer, loss_fn, device, mode, train: bool):
    model.train() if train else model.eval()
    total_loss = total_mae = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            img = batch["image"].to(device)
            meta = apply_focal_mode(batch["metadata"], mode, train).to(device)
            target_log = batch["target"].to(device)
            target_cm = batch["target_cm"].to(device)
            if train:
                optimizer.zero_grad()
            pred_log = model(img, meta)
            loss = loss_fn(pred_log, target_log)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_mae += mae_cm(pred_log.detach(), target_cm)
            n += 1
    n = max(n, 1)
    return {"loss": total_loss / n, "mae": total_mae / n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg = {
        "root_dir": args.root_dir, "image_size": 128,
        "batch_size": args.batch_size, "val_ratio": 0.15,
        "num_workers": args.num_workers, "seed": args.split_seed,
    }
    train_loader, val_loader = build_dataloaders(loader_cfg)
    print(f"focal_mode={args.focal_mode} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print("(val is ALWAYS scored under test-realistic const focal)")

    model = build_model().to(device)
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mae = float("inf")
    ckpt_path = out_dir / args.ckpt_name
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, loss_fn, device, args.focal_mode, True)
        va = run_epoch(model, val_loader, optimizer, loss_fn, device, args.focal_mode, False)
        scheduler.step()
        print(f"[{epoch:03d}/{args.epochs}] train_mae={tr['mae']:.2f}cm | "
              f"val_mae(const-focal)={va['mae']:.2f}cm ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train": tr, "val": va})
        if va["mae"] < best_mae:
            best_mae = va["mae"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best updated ({best_mae:.2f}cm), saved {ckpt_path}")

    with open(out_dir / f"history_{args.ckpt_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"DONE. best val MAE(const-focal) = {best_mae:.2f}cm -> {ckpt_path}")


if __name__ == "__main__":
    main()
