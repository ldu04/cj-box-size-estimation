# train.py
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="regression_dataset")
    p.add_argument("--labels_file", type=str, default="labels.json")
    p.add_argument("--image_size", type=int, default=128)  # B 코드 고정값
    p.add_argument("--batch_size", type=int, default=32)   # A100 환경 최적화
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, default="regressor_full_best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None,
                   help="train/val 분할 전용 seed (미지정 시 --seed 사용). "
                        "seed 앙상블 학습 시 분할은 고정하고 초기값만 바꾸는 용도")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def mae_cm(pred_log, target_cm):
    # 정직하게 자기 인덱스의 예측값과 자기 인덱스의 실제 cm 정답 간의 오차 계산
    return torch.mean(torch.abs(torch.exp(pred_log) - target_cm)).item()


@torch.no_grad()
def da3(pred_log, target_cm, base=1.25, k=3):
    pred_cm = torch.exp(pred_log).clamp(min=1e-3)
    tgt = target_cm.clamp(min=1e-3)
    ratio = torch.max(pred_cm / tgt, tgt / pred_cm)
    return (ratio < base ** k).float().mean().item()


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_mae = total_da3 = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            img = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            target_log = batch["target"].to(device)
            target_cm = batch["target_cm"].to(device)

            if train:
                optimizer.zero_grad()
            
            # 1:1 매칭 관계를 깨뜨리지 않고 정직하게 포워딩
            pred_log = model(img, meta)
            
            # 각 크롭 고유의 정답 로그값과 직접 Loss 비교
            loss = loss_fn(pred_log, target_log)
            
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_mae += mae_cm(pred_log.detach(), target_cm)
            total_da3 += da3(pred_log.detach(), target_cm)
            n += 1
    n = max(n, 1)
    return {"loss": total_loss / n, "mae": total_mae / n, "da3": total_da3 / n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg = {
        "root_dir": args.root_dir, "labels_file": args.labels_file,
        "image_size": args.image_size, "batch_size": args.batch_size,
        "val_ratio": args.val_ratio, "num_workers": args.num_workers,
        "seed": args.split_seed if args.split_seed is not None else args.seed,
    }
    train_loader, val_loader = build_dataloaders(loader_cfg)
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mae = float("inf")
    history = []
    ckpt_path = out_dir / args.ckpt_name

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_m = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        scheduler.step()

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_m['loss']:.4f} train_mae={train_m['mae']:.2f}cm | "
              f"val_loss={val_m['loss']:.4f} val_mae={val_m['mae']:.2f}cm val_da3={val_m['da3']:.3f} "
              f"({time.time()-t0:.1f}s)")

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best 갱신 (val_mae={best_mae:.2f}cm), {ckpt_path} 저장")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"학습 완료. best val MAE = {best_mae:.2f} cm, checkpoint = {ckpt_path}")


if __name__ == "__main__":
    main()
